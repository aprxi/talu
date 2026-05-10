use anyhow::{anyhow, Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use talu::{router, ChatHandle};

pub const VISION_PROFILE_VERSION: &str = "2026-04-17";
pub const RAW_IMAGE_PREPROCESS_REQUIRED_MESSAGE: &str =
    "input_image requires prepared vision input; raw image preprocessing is not served by talu";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PrepareVisionModelProfile {
    version: String,
    normalize: String,
    temporal_frames: u32,
    patch_size: u32,
    temporal_patch_size: u32,
    spatial_merge_size: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    smart_resize: Option<PrepareVisionSmartResize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    alpha_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    alpha_background: Option<PrepareVisionRgb>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PrepareVisionSmartResize {
    factor: u32,
    min_pixels: u64,
    max_pixels: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PrepareVisionRgb {
    r: u8,
    g: u8,
    b: u8,
}

#[derive(Debug, Clone, Deserialize)]
struct PreparedImage {
    #[serde(rename = "input_id")]
    #[serde(default)]
    _input_id: Option<String>,
    dtype: String,
    layout: String,
    channels: u32,
    temporal_frames: u32,
    width: u32,
    height: u32,
    grid: PreparedGrid,
    token_count: usize,
    normalize: String,
    tensor_b64: String,
}

#[derive(Debug, Clone, Deserialize)]
struct PreparedGrid {
    temporal: u32,
    height: u32,
    width: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparedImageEnvelope {
    model_profile: PrepareVisionModelProfile,
    item: PreparedImage,
}

impl Default for PrepareVisionModelProfile {
    fn default() -> Self {
        Self {
            version: VISION_PROFILE_VERSION.to_string(),
            normalize: "minus_one_to_one".to_string(),
            temporal_frames: 1,
            patch_size: 16,
            temporal_patch_size: 1,
            spatial_merge_size: 1,
            smart_resize: None,
            alpha_mode: None,
            alpha_background: None,
        }
    }
}

pub fn payload_may_include_images(input_json: Option<&str>, prev_json: Option<&str>) -> bool {
    [input_json, prev_json].into_iter().flatten().any(|s| {
        s.contains("\"input_image\"")
            || s.contains("\"image_url\"")
            || s.contains("\"prepared\"")
            || s.contains("\"type\":\"image\"")
            || s.contains("\"type\":\"image_url\"")
    })
}

pub fn stdin_json_has_prepared_payload(stdin_text: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(stdin_text) else {
        return false;
    };
    value
        .as_object()
        .and_then(|obj| obj.get("talu_prepared_vision"))
        .is_some()
}

pub fn prepare_vision_prefill_from_stdin_prepared_json(
    stdin_text: &str,
    model_id: &str,
) -> Result<Option<router::VisionPrefillInput>> {
    let value: Value =
        serde_json::from_str(stdin_text).context("decode stdin prepared vision JSON")?;
    let Some(root) = value.as_object() else {
        return Err(anyhow!(
            "stdin prepared vision payload must be a JSON object with 'talu_prepared_vision'"
        ));
    };
    let Some(payload) = root.get("talu_prepared_vision") else {
        return Ok(None);
    };

    let prepared_items = match payload {
        Value::Array(items) => {
            if items.is_empty() {
                return Err(anyhow!(
                    "'talu_prepared_vision' array must include at least one prepared item"
                ));
            }
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(parse_prepared_image_envelope(item)?);
            }
            out
        }
        Value::Object(_) => vec![parse_prepared_image_envelope(payload)?],
        _ => {
            return Err(anyhow!(
                "'talu_prepared_vision' must be an object or array of objects"
            ));
        }
    };

    build_prepared_prefill(model_id, 0, prepared_items, "stdin.talu_prepared_vision")
}

pub fn model_vision_preprocess_profile(model_id: &str) -> Option<Value> {
    if image_token_id_for_model(model_id).is_none() {
        return None;
    }
    serde_json::to_value(vision_profile_for_model(model_id)).ok()
}

pub fn validate_prepared_image_payload_shape(prepared: &Value) -> Result<()> {
    serde_json::from_value::<PreparedImageEnvelope>(prepared.clone())
        .context("prepared image must match {model_profile, item}")
        .map(|_| ())
}

pub fn extract_prepared_prefill_from_responses_input(
    input: &mut Value,
    model_id: &str,
) -> Result<Option<router::VisionPrefillInput>> {
    let Some(items) = input.as_array_mut() else {
        return Ok(None);
    };

    let mut raw_images = 0usize;
    let mut prepared_items = Vec::new();
    let mut prepared_index = 0usize;

    for item in items {
        let Some(item_obj) = item.as_object_mut() else {
            continue;
        };
        if item_obj.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let Some(parts) = item_obj.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for part in parts {
            let Some(part_obj) = part.as_object_mut() else {
                continue;
            };
            if part_obj.get("type").and_then(Value::as_str) != Some("input_image") {
                continue;
            }

            let prepared = part_obj.get("prepared").cloned();
            let image_url = part_obj
                .get("image_url")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(str::to_string);

            if let Some(prepared_value) = prepared {
                if image_url.is_some() {
                    return Err(anyhow!(
                        "input_image cannot include both image_url and prepared"
                    ));
                }
                let envelope = parse_prepared_image_envelope(&prepared_value)?;
                prepared_items.push(envelope);
                part_obj.insert(
                    "image_url".to_string(),
                    Value::String(format!("prepared://input_{prepared_index}")),
                );
                part_obj.remove("prepared");
                prepared_index += 1;
            } else if image_url.is_some() {
                raw_images += 1;
            }
        }
    }

    build_prepared_prefill(model_id, raw_images, prepared_items, "responses.input")
}

pub fn extract_prepared_prefill_from_completions_messages(
    messages: &mut [crate::server::completions_types::ChatMessage],
    model_id: &str,
) -> Result<Option<router::VisionPrefillInput>> {
    let mut raw_images = 0usize;
    let mut prepared_items = Vec::new();
    let mut prepared_index = 0usize;

    for message in messages {
        let Some(content) = message.content.as_mut() else {
            continue;
        };
        let Some(parts) = content.as_array_mut() else {
            continue;
        };
        for part in parts {
            let Some(part_obj) = part.as_object_mut() else {
                continue;
            };
            let part_type = part_obj
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if part_type != "image_url" && part_type != "image" {
                continue;
            }

            let prepared = part_obj.get("prepared").cloned();
            let has_raw_url = extract_chat_image_url(part_obj).is_some();

            if let Some(prepared_value) = prepared {
                if has_raw_url {
                    return Err(anyhow!(
                        "chat.completions image content cannot include both image_url and prepared"
                    ));
                }
                let envelope = parse_prepared_image_envelope(&prepared_value)?;
                prepared_items.push(envelope);
                part_obj.insert(
                    "image_url".to_string(),
                    serde_json::json!({
                        "url": format!("prepared://input_{prepared_index}")
                    }),
                );
                part_obj.remove("prepared");
                prepared_index += 1;
            } else if has_raw_url {
                raw_images += 1;
            }
        }
    }

    build_prepared_prefill(
        model_id,
        raw_images,
        prepared_items,
        "chat.completions.messages",
    )
}

pub fn prepare_vision_prefill(
    chat: &ChatHandle,
    model_id: &str,
) -> Result<Option<router::VisionPrefillInput>> {
    let image_urls = collect_input_image_urls(chat)?;
    prepare_vision_prefill_from_urls(model_id, image_urls)
}

pub fn prepare_vision_prefill_from_content(
    content: &[router::ContentPart],
    model_id: &str,
) -> Result<Option<router::VisionPrefillInput>> {
    let mut image_urls = Vec::new();
    for part in content {
        match part {
            router::ContentPart::ImageUrl { url, .. } => {
                let trimmed = url.trim();
                if !trimmed.is_empty() {
                    image_urls.push(trimmed.to_string());
                }
            }
            router::ContentPart::ImageBase64 { data, mime } => {
                image_urls.push(format!(
                    "data:{};base64,{}",
                    mime,
                    base64::engine::general_purpose::STANDARD.encode(data)
                ));
            }
            _ => {}
        }
    }
    prepare_vision_prefill_from_urls(model_id, image_urls)
}

fn prepare_vision_prefill_from_urls(
    model_id: &str,
    image_urls: Vec<String>,
) -> Result<Option<router::VisionPrefillInput>> {
    let _ = model_id;
    if image_urls.is_empty() {
        return Ok(None);
    }

    for image_url in &image_urls {
        validate_raw_image_source(image_url)?;
    }
    Err(anyhow!(RAW_IMAGE_PREPROCESS_REQUIRED_MESSAGE))
}

fn image_token_id_from_env() -> Option<u32> {
    if let Ok(raw) = std::env::var("TALU_IMAGE_TOKEN_ID") {
        if let Ok(parsed) = raw.trim().parse::<u32>() {
            if parsed > 0 {
                return Some(parsed);
            }
        }
    }
    None
}

fn image_token_id_for_model(model_id: &str) -> Option<u32> {
    let candidates = model_dir_candidates(model_id);
    for model_dir in candidates {
        let config_path = model_dir.join("config.json");
        let Ok(config_json) = std::fs::read_to_string(config_path) else {
            continue;
        };
        if let Some(token_id) = image_token_id_from_config_json(&config_json) {
            return Some(token_id);
        }
    }
    None
}

fn vision_profile_for_model(model_id: &str) -> PrepareVisionModelProfile {
    let mut profile = PrepareVisionModelProfile::default();
    let candidates = model_dir_candidates(model_id);
    for model_dir in candidates {
        let config_path = model_dir.join("config.json");
        if let Ok(config_json) = std::fs::read_to_string(config_path) {
            apply_profile_from_config_json(&mut profile, &config_json);
        }
        let processor_paths = [
            model_dir.join("processor_config.json"),
            model_dir.join("preprocessor_config.json"),
        ];
        for processor_path in processor_paths {
            if let Ok(processor_json) = std::fs::read_to_string(processor_path) {
                apply_profile_from_processor_json(&mut profile, &processor_json);
                normalize_temporal_profile(&mut profile);
                return profile;
            }
        }
    }
    normalize_temporal_profile(&mut profile);
    profile
}

fn model_dir_candidates(model_id: &str) -> Vec<PathBuf> {
    let candidates = [
        model_dir_candidate(model_id),
        talu::repo::resolve_model_path_ex(model_id, true)
            .ok()
            .map(PathBuf::from),
    ];
    candidates.into_iter().flatten().collect()
}

fn model_dir_candidate(model_id: &str) -> Option<PathBuf> {
    let path = PathBuf::from(model_id);
    if path.is_dir() {
        Some(path)
    } else {
        None
    }
}

fn image_token_id_from_config_json(config_json: &str) -> Option<u32> {
    let value: Value = serde_json::from_str(config_json).ok()?;
    value
        .get("image_token_id")
        .and_then(Value::as_u64)
        .or_else(|| value.get("image_token_index").and_then(Value::as_u64))
        .and_then(|v| u32::try_from(v).ok())
        .filter(|v| *v > 0)
}

fn vision_patch_size_from_config_json(config_json: &str) -> Option<u32> {
    let value: Value = serde_json::from_str(config_json).ok()?;
    value
        .get("vision_patch_size")
        .and_then(Value::as_u64)
        .or_else(|| {
            value
                .get("vision_config")
                .and_then(|v| v.get("patch_size"))
                .and_then(Value::as_u64)
        })
        .and_then(|v| u32::try_from(v).ok())
        .filter(|v| *v > 0)
}

fn vision_spatial_merge_size_from_config_json(config_json: &str) -> Option<u32> {
    let value: Value = serde_json::from_str(config_json).ok()?;
    value
        .get("vision_spatial_merge_size")
        .and_then(Value::as_u64)
        .or_else(|| {
            value
                .get("vision_config")
                .and_then(|v| v.get("spatial_merge_size"))
                .and_then(Value::as_u64)
        })
        .or_else(|| value.get("downsample_factor").and_then(Value::as_u64))
        .and_then(|v| u32::try_from(v).ok())
        .filter(|v| *v > 0)
}

fn vision_temporal_patch_size_from_config_json(config_json: &str) -> Option<u32> {
    let value: Value = serde_json::from_str(config_json).ok()?;
    value
        .get("vision_temporal_patch_size")
        .and_then(Value::as_u64)
        .or_else(|| {
            value
                .get("vision_config")
                .and_then(|v| v.get("temporal_patch_size"))
                .and_then(Value::as_u64)
        })
        .and_then(|v| u32::try_from(v).ok())
        .filter(|v| *v > 0)
}

fn apply_profile_from_config_json(profile: &mut PrepareVisionModelProfile, config_json: &str) {
    if let Some(patch_size) = vision_patch_size_from_config_json(config_json) {
        profile.patch_size = patch_size;
    }
    if let Some(spatial_merge_size) = vision_spatial_merge_size_from_config_json(config_json) {
        profile.spatial_merge_size = spatial_merge_size;
    }
    if let Some(temporal_patch_size) = vision_temporal_patch_size_from_config_json(config_json) {
        profile.temporal_patch_size = temporal_patch_size;
    }
    normalize_temporal_profile(profile);
}

fn apply_profile_from_processor_json(
    profile: &mut PrepareVisionModelProfile,
    processor_json: &str,
) {
    let Ok(value) = serde_json::from_str::<Value>(processor_json) else {
        return;
    };
    let proc = value.get("image_processor").unwrap_or(&value);

    if let Some(patch_size) =
        get_u32(proc, "encoder_patch_size").or_else(|| get_u32(proc, "patch_size"))
    {
        profile.patch_size = patch_size.max(1);
    }
    if let Some(spatial_merge_size) = get_u32(proc, "downsample_factor")
        .or_else(|| get_u32(proc, "spatial_merge_size"))
        .or_else(|| get_u32(proc, "merge_size"))
    {
        profile.spatial_merge_size = spatial_merge_size.max(1);
    }
    if let Some(temporal_patch_size) = get_u32(proc, "temporal_patch_size") {
        profile.temporal_patch_size = temporal_patch_size.max(1);
    }
    if let Some(temporal_frames) = get_u32(proc, "temporal_frames") {
        profile.temporal_frames = temporal_frames.max(1);
    }

    if let Some(normalize) = normalize_mode_from_processor(proc) {
        profile.normalize = normalize.to_string();
    }

    if let (Some(min_tokens), Some(max_tokens)) = (
        get_u64(proc, "min_image_tokens"),
        get_u64(proc, "max_image_tokens"),
    ) {
        let factor = profile
            .patch_size
            .saturating_mul(profile.spatial_merge_size)
            .max(1);
        if let Some(step_pixels) = checked_square_u64(factor) {
            let min_pixels = min_tokens.checked_mul(step_pixels).unwrap_or(0);
            let max_pixels = max_tokens.checked_mul(step_pixels).unwrap_or(0);
            if min_pixels > 0 && max_pixels >= min_pixels {
                profile.smart_resize = Some(PrepareVisionSmartResize {
                    factor,
                    min_pixels,
                    max_pixels,
                });
            }
        }
    }

    if profile.smart_resize.is_none() {
        let min_pixels = get_u64(proc, "min_pixels")
            .or_else(|| get_nested_u64(proc, &["size", "shortest_edge"]));
        let max_pixels =
            get_u64(proc, "max_pixels").or_else(|| get_nested_u64(proc, &["size", "longest_edge"]));
        if let (Some(min_pixels), Some(max_pixels)) = (min_pixels, max_pixels) {
            let factor = profile
                .patch_size
                .saturating_mul(profile.spatial_merge_size)
                .max(1);
            if min_pixels > 0 && max_pixels >= min_pixels {
                profile.smart_resize = Some(PrepareVisionSmartResize {
                    factor,
                    min_pixels,
                    max_pixels,
                });
            }
        }
    }

    normalize_temporal_profile(profile);
}

fn normalize_mode_from_processor(proc: &Value) -> Option<&'static str> {
    let do_rescale = proc
        .get("do_rescale")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let do_normalize = proc
        .get("do_normalize")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    if !do_rescale && !do_normalize {
        return Some("none");
    }
    if do_rescale && !do_normalize {
        return Some("zero_to_one");
    }

    let mean = proc.get("image_mean").and_then(as_f64_triplet);
    let std = proc.get("image_std").and_then(as_f64_triplet);
    if let (Some(mean), Some(std)) = (mean, std) {
        if approx_triplet(mean, [0.5, 0.5, 0.5]) && approx_triplet(std, [0.5, 0.5, 0.5]) {
            return Some("minus_one_to_one");
        }
        if approx_triplet(mean, [0.485, 0.456, 0.406]) && approx_triplet(std, [0.229, 0.224, 0.225])
        {
            return Some("imagenet");
        }
    }

    Some("minus_one_to_one")
}

fn as_f64_triplet(value: &Value) -> Option<[f64; 3]> {
    let arr = value.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    Some([arr[0].as_f64()?, arr[1].as_f64()?, arr[2].as_f64()?])
}

fn approx_triplet(left: [f64; 3], right: [f64; 3]) -> bool {
    const EPS: f64 = 1e-6;
    (left[0] - right[0]).abs() <= EPS
        && (left[1] - right[1]).abs() <= EPS
        && (left[2] - right[2]).abs() <= EPS
}

fn get_u32(obj: &Value, key: &str) -> Option<u32> {
    obj.get(key)
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
}

fn get_u64(obj: &Value, key: &str) -> Option<u64> {
    obj.get(key).and_then(Value::as_u64)
}

fn get_nested_u64(obj: &Value, path: &[&str]) -> Option<u64> {
    let mut current = obj;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_u64()
}

fn checked_square_u64(value: u32) -> Option<u64> {
    let v = u64::from(value);
    v.checked_mul(v)
}

fn normalize_temporal_profile(profile: &mut PrepareVisionModelProfile) {
    profile.patch_size = profile.patch_size.max(1);
    profile.spatial_merge_size = profile.spatial_merge_size.max(1);
    profile.temporal_patch_size = profile.temporal_patch_size.max(1);
    profile.temporal_frames = profile.temporal_frames.max(1);

    if profile.temporal_frames < profile.temporal_patch_size {
        profile.temporal_frames = profile.temporal_patch_size;
        return;
    }

    let rem = profile.temporal_frames % profile.temporal_patch_size;
    if rem != 0 {
        profile.temporal_frames += profile.temporal_patch_size - rem;
    }
}

fn validate_prepared_image_shape(
    img: &PreparedImage,
    expected_profile: &PrepareVisionModelProfile,
    model_id: &str,
    host: &str,
) -> Result<()> {
    if img.grid.temporal == 0 || img.grid.height == 0 || img.grid.width == 0 {
        return Err(anyhow!(
            "vision prepare response contains zero grid dimensions"
        ));
    }
    if img.channels != 3 {
        return Err(anyhow!(
            "vision prepare payload from {host} for model '{model_id}' must have channels=3, got {}",
            img.channels
        ));
    }
    if img.dtype != "f32" {
        return Err(anyhow!(
            "vision prepare payload from {host} for model '{model_id}' must have dtype=f32, got {}",
            img.dtype
        ));
    }
    if img.layout != "cthw" {
        return Err(anyhow!(
            "vision prepare payload from {host} for model '{model_id}' must have layout=cthw, got {}",
            img.layout
        ));
    }
    if img.normalize != expected_profile.normalize {
        return Err(anyhow!(
            "vision prepare payload from {host} for model '{model_id}' normalize mismatch: expected {}, got {}",
            expected_profile.normalize,
            img.normalize
        ));
    }
    if img.temporal_frames != expected_profile.temporal_frames {
        return Err(anyhow!(
            "vision prepare payload from {host} for model '{model_id}' temporal_frames mismatch: expected {}, got {}",
            expected_profile.temporal_frames,
            img.temporal_frames
        ));
    }

    let patch_size = expected_profile.patch_size.max(1);
    let expected_height =
        img.grid.height.checked_mul(patch_size).ok_or_else(|| {
            anyhow!("vision grid height overflows when validating prepare payload")
        })?;
    let expected_width =
        img.grid.width.checked_mul(patch_size).ok_or_else(|| {
            anyhow!("vision grid width overflows when validating prepare payload")
        })?;
    if img.height != expected_height || img.width != expected_width {
        return Err(anyhow!(
            "vision prepare payload from {host} does not match model '{model_id}' patch_size={patch_size}: got width={width}, height={height}, grid=({gt},{gh},{gw}), expected width={ew}, height={eh}",
            width = img.width,
            height = img.height,
            gt = img.grid.temporal,
            gh = img.grid.height,
            gw = img.grid.width,
            ew = expected_width,
            eh = expected_height,
        ));
    }

    let temporal_patch_size = expected_profile.temporal_patch_size.max(1);
    let expected_temporal = expected_profile
        .temporal_frames
        .checked_div(temporal_patch_size)
        .unwrap_or(0);
    if expected_temporal == 0 || img.grid.temporal != expected_temporal {
        return Err(anyhow!(
            "vision prepare payload from {host} does not match model '{model_id}' temporal grid: got {}, expected {}",
            img.grid.temporal,
            expected_temporal
        ));
    }

    let spatial_merge_size = expected_profile.spatial_merge_size.max(1);
    if img.grid.height % spatial_merge_size != 0 || img.grid.width % spatial_merge_size != 0 {
        return Err(anyhow!(
            "vision prepare payload from {host} has grid not divisible by spatial_merge_size={spatial_merge_size}: grid=({}, {})",
            img.grid.height,
            img.grid.width
        ));
    }
    let merged_h = img.grid.height / spatial_merge_size;
    let merged_w = img.grid.width / spatial_merge_size;
    let expected_tokens = u64::from(img.grid.temporal)
        .checked_mul(u64::from(merged_h))
        .and_then(|v| v.checked_mul(u64::from(merged_w)))
        .ok_or_else(|| anyhow!("vision token_count overflow while validating prepare payload"))?;
    if u64::try_from(img.token_count).ok() != Some(expected_tokens) {
        return Err(anyhow!(
            "vision prepare payload from {host} token_count mismatch for model '{model_id}': got {}, expected {}",
            img.token_count,
            expected_tokens
        ));
    }
    Ok(())
}

fn parse_prepared_image_envelope(value: &Value) -> Result<PreparedImageEnvelope> {
    serde_json::from_value::<PreparedImageEnvelope>(value.clone())
        .context("prepared image must include model_profile and item")
}

fn extract_chat_image_url(part_obj: &serde_json::Map<String, Value>) -> Option<String> {
    match part_obj.get("image_url") {
        Some(Value::String(url)) => {
            let trimmed = url.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        Some(Value::Object(obj)) => obj
            .get("url")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_string),
        _ => None,
    }
}

fn build_prepared_prefill(
    model_id: &str,
    raw_images: usize,
    prepared_items: Vec<PreparedImageEnvelope>,
    request_path: &str,
) -> Result<Option<router::VisionPrefillInput>> {
    if prepared_items.is_empty() {
        return Ok(None);
    }
    if raw_images > 0 {
        return Err(anyhow!(
            "{request_path} mixes prepared and raw images; use one image mode per request"
        ));
    }

    let image_token_id = image_token_id_from_env()
        .or_else(|| image_token_id_for_model(model_id))
        .ok_or_else(|| {
            anyhow!(
                "model '{model_id}' does not expose image_token_id; prepared vision input is unsupported"
            )
        })?;
    let expected_profile = vision_profile_for_model(model_id);
    let mut images = Vec::with_capacity(prepared_items.len());

    for (index, prepared) in prepared_items.into_iter().enumerate() {
        validate_profile_match(
            &prepared.model_profile,
            &expected_profile,
            model_id,
            request_path,
            index,
        )?;
        validate_prepared_image_shape(&prepared.item, &expected_profile, model_id, request_path)?;

        let pixels = decode_f32_base64(&prepared.item.tensor_b64)?;
        validate_tensor_pixel_count(&prepared.item, pixels.len(), request_path, index)?;

        images.push(router::VisionPrefillImage {
            pixels,
            width: prepared.item.width,
            height: prepared.item.height,
            grid_temporal: prepared.item.grid.temporal,
            grid_height: prepared.item.grid.height,
            grid_width: prepared.item.grid.width,
            token_count: prepared.item.token_count,
        });
    }

    Ok(Some(router::VisionPrefillInput {
        image_token_id,
        images,
    }))
}

fn validate_profile_match(
    provided: &PrepareVisionModelProfile,
    expected: &PrepareVisionModelProfile,
    model_id: &str,
    request_path: &str,
    index: usize,
) -> Result<()> {
    if provided != expected {
        return Err(anyhow!(
            "{request_path}[{index}] model_profile mismatch for model '{model_id}': expected {:?}, got {:?}",
            expected,
            provided
        ));
    }
    Ok(())
}

fn validate_tensor_pixel_count(
    image: &PreparedImage,
    pixels_len: usize,
    request_path: &str,
    index: usize,
) -> Result<()> {
    let expected = u64::from(image.channels)
        .checked_mul(u64::from(image.temporal_frames))
        .and_then(|v| v.checked_mul(u64::from(image.height)))
        .and_then(|v| v.checked_mul(u64::from(image.width)))
        .ok_or_else(|| anyhow!("prepared tensor size overflow"))?;
    let expected =
        usize::try_from(expected).map_err(|_| anyhow!("prepared tensor size overflow"))?;
    if pixels_len != expected {
        return Err(anyhow!(
            "{request_path}[{index}] tensor length mismatch: expected {expected} f32 values, got {pixels_len}"
        ));
    }
    Ok(())
}

fn collect_input_image_urls(chat: &ChatHandle) -> Result<Vec<String>> {
    let json = chat
        .to_responses_json(1)
        .map_err(|e| anyhow!("failed to serialize conversation for vision scan: {e}"))?;
    let value: Value =
        serde_json::from_str(&json).context("parse conversation JSON for vision scan")?;
    let items = value
        .as_array()
        .ok_or_else(|| anyhow!("conversation JSON must be an array"))?;

    let mut urls = Vec::new();
    for item in items {
        let Some("message") = item.get("type").and_then(Value::as_str) else {
            continue;
        };
        let Some(content) = item.get("content").and_then(Value::as_array) else {
            continue;
        };
        for part in content {
            if part.get("type").and_then(Value::as_str) != Some("input_image") {
                continue;
            }
            if let Some(url) = part.get("image_url").and_then(Value::as_str) {
                let trimmed = url.trim();
                if !trimmed.is_empty() {
                    urls.push(trimmed.to_string());
                }
            }
        }
    }
    Ok(urls)
}

fn decode_f32_base64(value: &str) -> Result<Vec<f32>> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(value)
        .context("decode tensor_b64")?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(anyhow!("tensor_b64 byte length is not a multiple of 4"));
    }

    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn validate_raw_image_source(image_url: &str) -> Result<()> {
    let trimmed = image_url.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("input_image URL cannot be empty"));
    }
    if trimmed.starts_with("data:")
        || trimmed.starts_with("http://")
        || trimmed.starts_with("https://")
        || ((trimmed.starts_with("file_") || trimmed.starts_with("file-"))
            && !trimmed.contains("://"))
    {
        return Ok(());
    }
    Err(anyhow!(
        "unsupported input_image source: expected data URL, http(s) URL, or file id"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn prepare_vision_prefill_from_urls_requires_prepared_input() {
        let data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jx9QAAAAASUVORK5CYII=";
        let err = prepare_vision_prefill_from_urls("test-model", vec![data_url.to_string()])
            .expect_err("raw image must fail");
        assert_eq!(err.to_string(), RAW_IMAGE_PREPROCESS_REQUIRED_MESSAGE);
    }

    #[test]
    fn prepare_vision_prefill_from_urls_rejects_file_id() {
        let err = prepare_vision_prefill_from_urls("test-model", vec!["file_abc".to_string()])
            .expect_err("file id must fail");
        assert_eq!(err.to_string(), RAW_IMAGE_PREPROCESS_REQUIRED_MESSAGE);
    }

    #[test]
    fn validate_raw_image_source_accepts_data_http_and_file_id() {
        validate_raw_image_source("file_abc").expect("file id");
        validate_raw_image_source("https://example.com/x.png").expect("http");
        validate_raw_image_source("data:image/png;base64,AA==").expect("data");
        let err = validate_raw_image_source("relative/path.png").expect_err("invalid");
        assert!(
            err.to_string().contains("unsupported input_image source"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn decode_f32_base64_roundtrip() {
        let bytes: Vec<u8> = vec![
            0x00, 0x00, 0x80, 0x3f, // 1.0
            0x00, 0x00, 0x00, 0x40, // 2.0
        ];
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        let decoded = decode_f32_base64(&encoded).expect("decode");
        assert_eq!(decoded, vec![1.0, 2.0]);
    }

    #[test]
    fn image_token_id_from_config_json_reads_primary_key() {
        let json = r#"{"image_token_id":151655}"#;
        assert_eq!(image_token_id_from_config_json(json), Some(151655));
    }

    #[test]
    fn image_token_id_from_config_json_reads_fallback_key() {
        let json = r#"{"image_token_index":1234}"#;
        assert_eq!(image_token_id_from_config_json(json), Some(1234));
    }

    #[test]
    fn vision_patch_size_from_config_json_reads_top_level_key() {
        let json = r#"{"vision_patch_size":16}"#;
        assert_eq!(vision_patch_size_from_config_json(json), Some(16));
    }

    #[test]
    fn vision_patch_size_from_config_json_reads_nested_key() {
        let json = r#"{"vision_config":{"patch_size":32}}"#;
        assert_eq!(vision_patch_size_from_config_json(json), Some(32));
    }

    #[test]
    fn apply_profile_from_processor_json_reads_lfm_style_fields() {
        let mut profile = PrepareVisionModelProfile::default();
        let json = r#"{
            "image_processor": {
                "encoder_patch_size": 16,
                "downsample_factor": 2,
                "do_rescale": true,
                "do_normalize": true,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "min_image_tokens": 64,
                "max_image_tokens": 256
            }
        }"#;
        apply_profile_from_processor_json(&mut profile, json);
        assert_eq!(profile.patch_size, 16);
        assert_eq!(profile.spatial_merge_size, 2);
        assert_eq!(profile.temporal_patch_size, 1);
        assert_eq!(profile.normalize, "minus_one_to_one");
        let smart = profile.smart_resize.expect("smart resize");
        assert_eq!(smart.factor, 32);
        assert_eq!(smart.min_pixels, 65536);
        assert_eq!(smart.max_pixels, 262144);
    }

    #[test]
    fn apply_profile_from_processor_json_reads_qwen_preprocessor_size_fields() {
        let mut profile = PrepareVisionModelProfile::default();
        let json = r#"{
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "size": {
                "shortest_edge": 65536,
                "longest_edge": 16777216
            },
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5]
        }"#;

        apply_profile_from_processor_json(&mut profile, json);
        assert_eq!(profile.patch_size, 16);
        assert_eq!(profile.spatial_merge_size, 2);
        assert_eq!(profile.temporal_patch_size, 2);
        assert_eq!(profile.temporal_frames, 2);
        let smart = profile.smart_resize.expect("smart resize");
        assert_eq!(smart.factor, 32);
        assert_eq!(smart.min_pixels, 65536);
        assert_eq!(smart.max_pixels, 16777216);
    }

    #[test]
    fn apply_profile_from_config_json_normalizes_temporal_frames() {
        let mut profile = PrepareVisionModelProfile::default();
        apply_profile_from_config_json(
            &mut profile,
            r#"{"vision_config":{"temporal_patch_size":2}}"#,
        );
        assert_eq!(profile.temporal_patch_size, 2);
        assert_eq!(profile.temporal_frames, 2);
    }

    #[test]
    fn validate_prepared_image_shape_rejects_patch_mismatch() {
        let profile = PrepareVisionModelProfile {
            version: VISION_PROFILE_VERSION.to_string(),
            patch_size: 16,
            temporal_frames: 1,
            temporal_patch_size: 1,
            spatial_merge_size: 2,
            normalize: "minus_one_to_one".to_string(),
            smart_resize: None,
            alpha_mode: None,
            alpha_background: None,
        };
        let img = PreparedImage {
            _input_id: Some("input_0".to_string()),
            dtype: "f32".to_string(),
            layout: "cthw".to_string(),
            channels: 3,
            temporal_frames: 1,
            width: 56,
            height: 56,
            grid: PreparedGrid {
                temporal: 1,
                height: 4,
                width: 4,
            },
            token_count: 4,
            normalize: "minus_one_to_one".to_string(),
            tensor_b64: String::new(),
        };
        let err = validate_prepared_image_shape(&img, &profile, "model-x", "http://localhost")
            .expect_err("must reject patch mismatch");
        assert!(err.to_string().contains("patch_size=16"));
    }

    #[test]
    fn extract_prepared_prefill_from_responses_input_uses_prepared_payload() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        std::env::set_var("TALU_IMAGE_TOKEN_ID", "151655");

        let pixels = vec![0f32; 3 * 16 * 16];
        let mut raw = Vec::with_capacity(pixels.len() * 4);
        for value in pixels {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        let tensor_b64 = base64::engine::general_purpose::STANDARD.encode(raw);

        let mut input = serde_json::json!([
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "prepared": {
                            "model_profile": {
                                "version": VISION_PROFILE_VERSION,
                                "normalize": "minus_one_to_one",
                                "temporal_frames": 1,
                                "patch_size": 16,
                                "temporal_patch_size": 1,
                                "spatial_merge_size": 1
                            },
                            "item": {
                                "dtype": "f32",
                                "layout": "cthw",
                                "channels": 3,
                                "temporal_frames": 1,
                                "width": 16,
                                "height": 16,
                                "grid": { "temporal": 1, "height": 1, "width": 1 },
                                "token_count": 1,
                                "normalize": "minus_one_to_one",
                                "tensor_b64": tensor_b64
                            }
                        }
                    }
                ]
            }
        ]);

        let prefill = extract_prepared_prefill_from_responses_input(&mut input, "test-model")
            .expect("prefill");
        let prefill = prefill.expect("prepared prefill");
        assert_eq!(prefill.image_token_id, 151655);
        assert_eq!(prefill.images.len(), 1);
        assert_eq!(prefill.images[0].token_count, 1);

        let part = &input[0]["content"][0];
        assert_eq!(part["type"], "input_image");
        assert_eq!(part["image_url"], "prepared://input_0");
        assert!(part.get("prepared").is_none());

        std::env::remove_var("TALU_IMAGE_TOKEN_ID");
    }

    #[test]
    fn extract_prepared_prefill_from_responses_input_rejects_mixed_modes() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        std::env::set_var("TALU_IMAGE_TOKEN_ID", "151655");

        let mut input = serde_json::json!([
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "prepared": {
                            "model_profile": {
                                "version": VISION_PROFILE_VERSION,
                                "normalize": "minus_one_to_one",
                                "temporal_frames": 1,
                                "patch_size": 16,
                                "temporal_patch_size": 1,
                                "spatial_merge_size": 1
                            },
                            "item": {
                                "dtype": "f32",
                                "layout": "cthw",
                                "channels": 3,
                                "temporal_frames": 1,
                                "width": 16,
                                "height": 16,
                                "grid": { "temporal": 1, "height": 1, "width": 1 },
                                "token_count": 1,
                                "normalize": "minus_one_to_one",
                                "tensor_b64": base64::engine::general_purpose::STANDARD
                                    .encode(vec![0u8; 3 * 16 * 16 * 4])
                            }
                        }
                    },
                    {
                        "type": "input_image",
                        "image_url": "file_123"
                    }
                ]
            }
        ]);

        let err = extract_prepared_prefill_from_responses_input(&mut input, "test-model")
            .expect_err("mixed modes must fail");
        assert!(err.to_string().contains("mixes prepared and raw images"));
        std::env::remove_var("TALU_IMAGE_TOKEN_ID");
    }

    #[test]
    fn stdin_json_has_prepared_payload_detects_wrapper() {
        let yes = r#"{"talu_prepared_vision":{"model_profile":{},"item":{}}}"#;
        let no = r#"{"input":"hello"}"#;
        assert!(stdin_json_has_prepared_payload(yes));
        assert!(!stdin_json_has_prepared_payload(no));
    }

    #[test]
    fn prepare_vision_prefill_from_stdin_prepared_json_supports_wrapper() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        std::env::set_var("TALU_IMAGE_TOKEN_ID", "151655");

        let payload = serde_json::json!({
            "talu_prepared_vision": {
                "model_profile": {
                    "version": VISION_PROFILE_VERSION,
                    "normalize": "minus_one_to_one",
                    "temporal_frames": 1,
                    "patch_size": 16,
                    "temporal_patch_size": 1,
                    "spatial_merge_size": 1
                },
                "item": {
                    "dtype": "f32",
                    "layout": "cthw",
                    "channels": 3,
                    "temporal_frames": 1,
                    "width": 16,
                    "height": 16,
                    "grid": { "temporal": 1, "height": 1, "width": 1 },
                    "token_count": 1,
                    "normalize": "minus_one_to_one",
                    "tensor_b64": base64::engine::general_purpose::STANDARD
                        .encode(vec![0u8; 3 * 16 * 16 * 4])
                }
            }
        });
        let raw = serde_json::to_string(&payload).expect("encode payload");
        let prefill = prepare_vision_prefill_from_stdin_prepared_json(&raw, "test-model")
            .expect("prefill parse")
            .expect("prefill exists");
        assert_eq!(prefill.image_token_id, 151655);
        assert_eq!(prefill.images.len(), 1);
        assert_eq!(prefill.images[0].token_count, 1);
        std::env::remove_var("TALU_IMAGE_TOKEN_ID");
    }

    #[test]
    fn model_vision_preprocess_profile_includes_version() {
        let temp = tempfile::tempdir().expect("tempdir");
        let model_dir = temp.path();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{
                "image_token_id": 151655,
                "vision_config": {
                    "patch_size": 16,
                    "spatial_merge_size": 2
                }
            }"#,
        )
        .expect("write config");
        std::fs::write(
            model_dir.join("processor_config.json"),
            r#"{
                "image_processor": {
                    "do_rescale": true,
                    "do_normalize": true,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5]
                }
            }"#,
        )
        .expect("write processor");

        let profile =
            model_vision_preprocess_profile(model_dir.to_string_lossy().as_ref()).expect("profile");
        assert_eq!(profile["version"].as_str(), Some(VISION_PROFILE_VERSION));
        assert_eq!(profile["patch_size"].as_u64(), Some(16));
        assert_eq!(profile["spatial_merge_size"].as_u64(), Some(2));
    }
}
