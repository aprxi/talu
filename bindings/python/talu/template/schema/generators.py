"""Prompt generation strategies for different model types.

Models vary wildly in how they prefer to receive schema instructions:
- Coding models love TypeScript
- Older models prefer JSON Schema
- Instruction-tuned models often prefer XML-wrapped definitions
"""

from __future__ import annotations

import json
from typing import Any, Protocol


class PromptGenerator(Protocol):
    """Protocol for prompt generation strategies."""

    def generate(
        self,
        schema: dict[str, Any],
        allow_thinking: bool = False,
    ) -> str:
        """Generate a prompt description from a JSON Schema."""
        ...


class TypeScriptGenerator:
    """TypeScript interface syntax strategy (default)."""

    INTERFACE_PREFIX = "I_"

    def generate(
        self,
        schema: dict[str, Any],
        allow_thinking: bool = False,
    ) -> str:
        """Generate TypeScript interface from JSON schema."""
        lines = []

        if allow_thinking:
            lines.extend(
                [
                    "First, analyze the request inside <think></think> tags.",
                    "Close the </think> tag before the final response.",
                    "Then output your response as JSON with no extra text.",
                    "",
                ]
            )

        lines.append("Respond with JSON matching this schema:")
        lines.append("Output a single-line JSON response with no extra whitespace.")
        lines.append("Use the exact values requested by the user.")
        lines.append("```typescript")

        defs = schema.get("$defs", {})
        emitted: set[str] = set()

        for def_name, def_schema in defs.items():
            lines.extend(self._emit_interface(def_name, def_schema, defs, emitted))
            lines.append("")

        if "anyOf" in schema or "oneOf" in schema:
            lines.extend(self._emit_root_union(schema, defs, emitted))
        elif schema.get("type") == "array":
            root_type = self._resolve_type(schema, defs)
            lines.append(f"type Response = {root_type};")
        else:
            lines.extend(self._emit_interface("Response", schema, defs, emitted))

        lines.append("```")
        return "\n".join(lines)

    def _emit_root_union(
        self,
        schema: dict[str, Any],
        defs: dict[str, Any],
        emitted: set[str],
    ) -> list[str]:
        """Emit root-level union type (anyOf/oneOf) with discriminator support."""
        lines = []
        union_key = "anyOf" if "anyOf" in schema else "oneOf"
        options = schema[union_key]

        discriminator = schema.get("discriminator", {})
        discriminator_field = discriminator.get("propertyName")

        if discriminator_field:
            lines.append(
                f"// Discriminated union: use '{discriminator_field}' field to determine type"
            )
            lines.append("")

        type_names = []
        for i, option in enumerate(options):
            name = self._infer_type_name(option, i)
            type_names.append(name)

            if option.get("type") == "object" or "properties" in option:
                lines.extend(
                    self._emit_interface_with_discriminator(
                        name, option, defs, emitted, discriminator_field
                    )
                )
                lines.append("")
            else:
                if name not in emitted:
                    emitted.add(name)
                    lines.append(f"type {name} = {self._resolve_type(option, defs)};")
                    lines.append("")

        union_types = " | ".join(type_names)
        lines.append(f"type Response = {union_types};")

        return lines

    def _emit_interface_with_discriminator(
        self,
        name: str,
        schema: dict[str, Any],
        defs: dict[str, Any],
        emitted: set[str],
        discriminator_field: str | None,
    ) -> list[str]:
        if name in emitted:
            return []
        emitted.add(name)

        lines = [f"interface {name} {{"]

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        if discriminator_field and discriminator_field in properties:
            prop_schema = properties[discriminator_field]
            type_str = self._resolve_type(prop_schema, defs)

            if "const" in prop_schema:
                type_str = f'"{prop_schema["const"]}"'
            elif "enum" in prop_schema and len(prop_schema["enum"]) == 1:
                type_str = f'"{prop_schema["enum"][0]}"'

            lines.append(f"  {discriminator_field}: {type_str};  // <- Discriminator")

        for prop_name, prop_schema in properties.items():
            if prop_name == discriminator_field:
                continue

            type_str = self._resolve_type(prop_schema, defs)
            is_required = prop_name in required
            optional = "" if is_required else "?"

            description = prop_schema.get("description", "")
            if description:
                description = description.replace("\n", " ").strip()
            comment = f"  // {description}" if description else ""

            lines.append(f"  {prop_name}{optional}: {type_str};{comment}")

        lines.append("}")
        return lines

    def _infer_type_name(self, schema: dict[str, Any], index: int) -> str:
        if "title" in schema:
            return self._sanitize_name(schema["title"])
        if "$ref" in schema:
            return schema["$ref"].split("/")[-1]
        return f"Option{index + 1}"

    def _sanitize_name(self, name: str) -> str:
        sanitized = "".join(c if c.isalnum() else "_" for c in name)
        if sanitized and not sanitized[0].isalpha():
            sanitized = "T_" + sanitized
        return sanitized or "Unknown"

    def _emit_interface(
        self,
        name: str,
        schema: dict[str, Any],
        defs: dict[str, Any],
        emitted: set[str],
    ) -> list[str]:
        if name in emitted:
            return []
        emitted.add(name)

        lines = [f"interface {name} {{"]

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for prop_name, prop_schema in properties.items():
            type_str = self._resolve_type(prop_schema, defs)
            is_required = prop_name in required
            optional = "" if is_required else "?"

            description = prop_schema.get("description", "")
            comment = f"  // {description}" if description else ""

            lines.append(f"  {prop_name}{optional}: {type_str};{comment}")

        lines.append("}")
        return lines

    def _resolve_type(self, prop: dict[str, Any], defs: dict[str, Any]) -> str:
        if "$ref" in prop:
            ref_path = prop["$ref"]
            return ref_path.split("/")[-1]

        if "anyOf" in prop:
            types = []
            for option in prop["anyOf"]:
                if option.get("type") == "null":
                    types.append("null")
                else:
                    types.append(self._resolve_type(option, defs))
            if "null" in types:
                types = [t for t in types if t != "null"] + ["null"]
            return " | ".join(types)

        if "oneOf" in prop:
            types = [self._resolve_type(option, defs) for option in prop["oneOf"]]
            return " | ".join(types)

        if "enum" in prop:
            return " | ".join(f'"{v}"' for v in prop["enum"])

        if "const" in prop:
            v = prop["const"]
            return f'"{v}"' if isinstance(v, str) else str(v)

        type_val = prop.get("type")

        if type_val == "string":
            return "string"
        if type_val in ("number", "integer"):
            return "number"
        if type_val == "boolean":
            return "boolean"
        if type_val == "null":
            return "null"
        if type_val == "array":
            items = prop.get("items", {})
            item_type = self._resolve_type(items, defs)
            return f"{item_type}[]"
        if type_val == "object":
            return "object"

        return "any"


class JsonSchemaGenerator:
    """JSON Schema dump strategy for older models."""

    def generate(
        self,
        schema: dict[str, Any],
        allow_thinking: bool = False,
    ) -> str:
        """Generate JSON schema dump string."""
        lines = []

        if allow_thinking:
            lines.extend(
                [
                    "First, analyze the request inside <think></think> tags.",
                    "Close the </think> tag before the final response.",
                    "Then output your response as JSON with no extra text.",
                    "",
                ]
            )

        lines.append("Respond with JSON matching this schema:")
        lines.append("Output a single-line JSON response with no extra whitespace.")
        lines.append("Use the exact values requested by the user.")
        lines.append("```json")
        lines.append(json.dumps(schema, indent=2))
        lines.append("```")

        return "\n".join(lines)


class XmlGenerator:
    """XML-wrapped schema strategy for Anthropic-style models."""

    def generate(
        self,
        schema: dict[str, Any],
        allow_thinking: bool = False,
    ) -> str:
        """Generate XML-wrapped schema string."""
        lines = []

        if allow_thinking:
            lines.extend(
                [
                    "First, analyze the request inside <think></think> tags.",
                    "Close the </think> tag before the final response.",
                    "Then output your response as JSON with no extra text.",
                    "",
                ]
            )

        lines.append("Respond with JSON matching this schema:")
        lines.append("Output a single-line JSON response with no extra whitespace.")
        lines.append("Use the exact values requested by the user.")
        lines.append("<schema>")
        lines.extend(self._schema_to_xml(schema, indent=2))
        lines.append("</schema>")

        return "\n".join(lines)

    def _schema_to_xml(
        self,
        schema: dict[str, Any],
        indent: int = 0,
    ) -> list[str]:
        lines = []
        pad = " " * indent

        if "properties" in schema:
            for name, prop in schema["properties"].items():
                required = name in schema.get("required", [])
                req_attr = ' required="true"' if required else ""
                type_str = prop.get("type", "any")
                desc = prop.get("description", "")

                if "enum" in prop:
                    enum_str = ", ".join(f'"{v}"' for v in prop["enum"])
                    lines.append(f'{pad}<field name="{name}" type="enum({enum_str})"{req_attr}/>')
                elif type_str == "array":
                    item_type = prop.get("items", {}).get("type", "any")
                    lines.append(f'{pad}<field name="{name}" type="array<{item_type}>"{req_attr}/>')
                else:
                    comment = f"  <!-- {desc} -->" if desc else ""
                    lines.append(
                        f'{pad}<field name="{name}" type="{type_str}"{req_attr}/>{comment}'
                    )

        return lines
