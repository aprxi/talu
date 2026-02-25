use serde_json::Value;
use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;

const SUPPORTED_SCHEMA_KEYWORDS: &[&str] = &[
    "$ref",
    "type",
    "properties",
    "required",
    "allOf",
    "anyOf",
    "oneOf",
    "enum",
    "items",
    "additionalProperties",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minProperties",
    "maxProperties",
    "minItems",
    "maxItems",
    "discriminator",
    "description",
    "title",
    "default",
    "example",
    "examples",
    "format",
    "x-enumDescriptions",
    "x-unionTitle",
    "x-oaiTypeLabel",
    "x-unionDisplay",
    "pattern",
];

pub struct OpenApiSchemaValidator {
    spec: Value,
}

impl OpenApiSchemaValidator {
    pub fn load_responses_spec() -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../issues/responses-openapi.json")
            .canonicalize()
            .expect("canonicalize responses-openapi.json path");
        let raw = std::fs::read_to_string(path).expect("read responses-openapi.json");
        let spec: Value = serde_json::from_str(&raw).expect("parse responses-openapi.json");
        Self { spec }
    }

    pub fn spec(&self) -> &Value {
        &self.spec
    }

    pub fn schema_by_name(&self, name: &str) -> &Value {
        &self.spec["components"]["schemas"][name]
    }

    pub fn validate_named_schema(
        &self,
        schema_name: &str,
        value: &Value,
    ) -> Result<(), Vec<String>> {
        self.validate_schema(self.schema_by_name(schema_name), value)
    }

    pub fn validate_schema(&self, schema: &Value, value: &Value) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        self.validate_inner(schema, value, "$", &mut errors, 0);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn validate_responses_stream_event_schema(&self, event: &Value) -> Result<(), Vec<String>> {
        let schema = self
            .spec
            .pointer("/paths/~1responses/post/responses/200/content/text~1event-stream/schema")
            .expect("responses stream schema");
        self.validate_schema(schema, event)
    }

    pub fn unsupported_keywords_for_responses_surface(&self) -> BTreeSet<String> {
        let mut keywords = BTreeSet::new();
        let mut seen_refs = HashSet::new();

        let roots = [
            "/paths/~1responses/post/requestBody/content/application~1json/schema",
            "/paths/~1responses/post/responses/200/content/application~1json/schema",
            "/paths/~1responses/post/responses/200/content/text~1event-stream/schema",
        ];

        for ptr in roots {
            let schema = self
                .spec
                .pointer(ptr)
                .unwrap_or_else(|| panic!("missing schema pointer {ptr}"));
            self.walk_schema(schema, &mut keywords, &mut seen_refs, 0);
        }

        let allowed: BTreeSet<String> = SUPPORTED_SCHEMA_KEYWORDS
            .iter()
            .map(|k| k.to_string())
            .collect();
        keywords
            .difference(&allowed)
            .cloned()
            .collect::<BTreeSet<String>>()
    }

    fn walk_schema(
        &self,
        schema: &Value,
        out: &mut BTreeSet<String>,
        seen_refs: &mut HashSet<String>,
        depth: usize,
    ) {
        if depth > 256 {
            return;
        }
        let Some(obj) = schema.as_object() else {
            return;
        };

        for key in obj.keys() {
            out.insert(key.clone());
        }

        if let Some(reference) = obj.get("$ref").and_then(|v| v.as_str()) {
            if seen_refs.insert(reference.to_string()) {
                if let Some(target) = self.resolve_ref(reference) {
                    self.walk_schema(target, out, seen_refs, depth + 1);
                }
            }
        }

        for key in ["allOf", "anyOf", "oneOf"] {
            if let Some(items) = obj.get(key).and_then(|v| v.as_array()) {
                for item in items {
                    self.walk_schema(item, out, seen_refs, depth + 1);
                }
            }
        }
        if let Some(properties) = obj.get("properties").and_then(|v| v.as_object()) {
            for subschema in properties.values() {
                self.walk_schema(subschema, out, seen_refs, depth + 1);
            }
        }
        if let Some(items) = obj.get("items") {
            self.walk_schema(items, out, seen_refs, depth + 1);
        }
        if let Some(additional) = obj.get("additionalProperties") {
            if additional.is_object() {
                self.walk_schema(additional, out, seen_refs, depth + 1);
            }
        }
    }

    fn validate_inner(
        &self,
        schema: &Value,
        value: &Value,
        path: &str,
        errors: &mut Vec<String>,
        depth: usize,
    ) {
        if depth > 256 {
            errors.push(format!("{path}: schema recursion depth exceeded"));
            return;
        }
        let Some(obj) = schema.as_object() else {
            return;
        };

        if let Some(reference) = obj.get("$ref").and_then(|v| v.as_str()) {
            if let Some(target) = self.resolve_ref(reference) {
                self.validate_inner(target, value, path, errors, depth + 1);
            } else {
                errors.push(format!("{path}: unresolved $ref `{reference}`"));
                return;
            }
        }

        if let Some(all_of) = obj.get("allOf").and_then(|v| v.as_array()) {
            for child in all_of {
                self.validate_inner(child, value, path, errors, depth + 1);
            }
        }

        if let Some(any_of) = obj.get("anyOf").and_then(|v| v.as_array()) {
            let mut any_ok = false;
            for child in any_of {
                if self.schema_matches(child, value, depth + 1) {
                    any_ok = true;
                    break;
                }
            }
            if !any_ok {
                errors.push(format!("{path}: value did not match anyOf"));
            }
        }

        if let Some(one_of) = obj.get("oneOf").and_then(|v| v.as_array()) {
            let mut matches = 0usize;
            for child in one_of {
                if self.schema_matches(child, value, depth + 1) {
                    matches += 1;
                }
            }
            if matches != 1 {
                errors.push(format!("{path}: value matched {matches} oneOf schemas"));
            }
        }

        if let Some(enum_values) = obj.get("enum").and_then(|v| v.as_array()) {
            let found = enum_values.iter().any(|entry| entry == value);
            if !found {
                errors.push(format!("{path}: value {value} not in enum"));
            }
        }

        if let Some(expected_type) = obj.get("type").and_then(|v| v.as_str()) {
            let ok = match expected_type {
                "string" => value.is_string(),
                "number" => value.is_number(),
                "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
                "boolean" => value.is_boolean(),
                "object" => value.is_object(),
                "array" => value.is_array(),
                "null" => value.is_null(),
                _ => true,
            };
            if !ok {
                errors.push(format!(
                    "{path}: expected type `{expected_type}`, got {}",
                    self.value_type_name(value)
                ));
            }
        }

        if let Some(min) = obj.get("minimum").and_then(|v| v.as_f64()) {
            if let Some(num) = value.as_f64() {
                if num < min {
                    errors.push(format!("{path}: number {num} below minimum {min}"));
                }
            }
        }
        if let Some(max) = obj.get("maximum").and_then(|v| v.as_f64()) {
            if let Some(num) = value.as_f64() {
                if num > max {
                    errors.push(format!("{path}: number {num} above maximum {max}"));
                }
            }
        }

        if let Some(min_len) = obj.get("minLength").and_then(|v| v.as_u64()) {
            if let Some(s) = value.as_str() {
                let len = s.chars().count() as u64;
                if len < min_len {
                    errors.push(format!(
                        "{path}: string length {len} below minLength {min_len}"
                    ));
                }
            }
        }
        if let Some(max_len) = obj.get("maxLength").and_then(|v| v.as_u64()) {
            if let Some(s) = value.as_str() {
                let len = s.chars().count() as u64;
                if len > max_len {
                    errors.push(format!(
                        "{path}: string length {len} above maxLength {max_len}"
                    ));
                }
            }
        }
        if let Some(pattern) = obj.get("pattern").and_then(|v| v.as_str()) {
            if let Some(s) = value.as_str() {
                if !self.matches_pattern(pattern, s) {
                    errors.push(format!(
                        "{path}: string value `{s}` does not match pattern `{pattern}`"
                    ));
                }
            }
        }

        if let Some(min_props) = obj.get("minProperties").and_then(|v| v.as_u64()) {
            if let Some(map) = value.as_object() {
                let len = map.len() as u64;
                if len < min_props {
                    errors.push(format!(
                        "{path}: object size {len} below minProperties {min_props}"
                    ));
                }
            }
        }
        if let Some(max_props) = obj.get("maxProperties").and_then(|v| v.as_u64()) {
            if let Some(map) = value.as_object() {
                let len = map.len() as u64;
                if len > max_props {
                    errors.push(format!(
                        "{path}: object size {len} above maxProperties {max_props}"
                    ));
                }
            }
        }

        if let Some(min_items) = obj.get("minItems").and_then(|v| v.as_u64()) {
            if let Some(items) = value.as_array() {
                let len = items.len() as u64;
                if len < min_items {
                    errors.push(format!(
                        "{path}: array length {len} below minItems {min_items}"
                    ));
                }
            }
        }
        if let Some(max_items) = obj.get("maxItems").and_then(|v| v.as_u64()) {
            if let Some(items) = value.as_array() {
                let len = items.len() as u64;
                if len > max_items {
                    errors.push(format!(
                        "{path}: array length {len} above maxItems {max_items}"
                    ));
                }
            }
        }

        if let Some(required) = obj.get("required").and_then(|v| v.as_array()) {
            if let Some(map) = value.as_object() {
                for key in required {
                    if let Some(key) = key.as_str() {
                        if !map.contains_key(key) {
                            errors.push(format!("{path}: missing required property `{key}`"));
                        }
                    }
                }
            }
        }

        if let Some(map) = value.as_object() {
            if let Some(properties) = obj.get("properties").and_then(|v| v.as_object()) {
                for (key, prop_schema) in properties {
                    if let Some(prop_value) = map.get(key) {
                        let child_path = format!("{path}.{key}");
                        self.validate_inner(
                            prop_schema,
                            prop_value,
                            &child_path,
                            errors,
                            depth + 1,
                        );
                    }
                }

                if let Some(additional) = obj.get("additionalProperties") {
                    match additional {
                        Value::Bool(false) => {
                            for key in map.keys() {
                                if !properties.contains_key(key) {
                                    errors.push(format!(
                                        "{path}: additional property `{key}` is not allowed"
                                    ));
                                }
                            }
                        }
                        Value::Bool(true) => {}
                        _ => {
                            for (key, v) in map {
                                if !properties.contains_key(key) {
                                    let child_path = format!("{path}.{key}");
                                    self.validate_inner(
                                        additional,
                                        v,
                                        &child_path,
                                        errors,
                                        depth + 1,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            if let Some(additional) = obj.get("additionalProperties") {
                let properties = obj.get("properties").and_then(|v| v.as_object());
                match additional {
                    Value::Bool(false) => {
                        for key in map.keys() {
                            let is_additional = match properties {
                                Some(p) => !p.contains_key(key),
                                None => true,
                            };
                            if is_additional {
                                errors.push(format!(
                                    "{path}: additional property `{key}` is not allowed"
                                ));
                            }
                        }
                    }
                    Value::Bool(true) => {}
                    _ => {
                        for (key, v) in map {
                            let is_additional = match properties {
                                Some(p) => !p.contains_key(key),
                                None => true,
                            };
                            if is_additional {
                                let child_path = format!("{path}.{key}");
                                self.validate_inner(additional, v, &child_path, errors, depth + 1);
                            }
                        }
                    }
                }
            }
        }

        if let Some(items_schema) = obj.get("items") {
            if let Some(items) = value.as_array() {
                for (idx, item) in items.iter().enumerate() {
                    let child_path = format!("{path}[{idx}]");
                    self.validate_inner(items_schema, item, &child_path, errors, depth + 1);
                }
            }
        }
    }

    fn schema_matches(&self, schema: &Value, value: &Value, depth: usize) -> bool {
        let mut errs = Vec::new();
        self.validate_inner(schema, value, "$", &mut errs, depth);
        errs.is_empty()
    }

    fn resolve_ref(&self, reference: &str) -> Option<&Value> {
        let ptr = reference.strip_prefix('#')?;
        self.spec.pointer(ptr)
    }

    fn value_type_name(&self, v: &Value) -> &'static str {
        if v.is_null() {
            "null"
        } else if v.is_boolean() {
            "boolean"
        } else if v.as_i64().is_some() || v.as_u64().is_some() {
            "integer"
        } else if v.is_number() {
            "number"
        } else if v.is_string() {
            "string"
        } else if v.is_array() {
            "array"
        } else {
            "object"
        }
    }

    fn matches_pattern(&self, pattern: &str, s: &str) -> bool {
        match pattern {
            // Current OpenResponses spec pattern used for JSON-schema format names.
            "^[a-zA-Z0-9_-]+$" => {
                !s.is_empty()
                    && s.chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            }
            _ => false,
        }
    }
}
