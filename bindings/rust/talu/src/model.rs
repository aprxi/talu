//! Safe wrappers for talu model information and utilities.

use crate::error::error_from_last_or;
use crate::Result;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;

/// Quantization method used in a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMethod {
    /// No quantization (F16/F32)
    None,
    /// Grouped Affine quantization (MLX compatible)
    Gaffine,
    /// MXFP4 quantization
    Mxfp4,
    /// Native format (reserved, not currently used)
    Native,
}

impl From<talu_sys::QuantMethodEnum> for QuantMethod {
    fn from(m: talu_sys::QuantMethodEnum) -> Self {
        match m {
            talu_sys::QuantMethodEnum::None => QuantMethod::None,
            talu_sys::QuantMethodEnum::Gaffine => QuantMethod::Gaffine,
            talu_sys::QuantMethodEnum::Mxfp4 => QuantMethod::Mxfp4,
            talu_sys::QuantMethodEnum::Native => QuantMethod::Native,
        }
    }
}

/// Model architecture information.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub intermediate_size: i32,
    pub max_seq_len: i32,
    pub head_dim: i32,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub quant_method: QuantMethod,
    pub quant_bits: i32,
    pub quant_group_size: i32,
    pub model_type: String,
    pub architecture: String,
    pub tie_word_embeddings: bool,
    pub use_gelu: bool,
    pub num_experts: i32,
    pub experts_per_token: i32,
}

/// Generation configuration from model config.
#[derive(Debug, Clone, Default)]
pub struct GenerationConfigInfo {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub do_sample: bool,
}

/// Describes a model, returning its architecture information.
pub fn describe(model_path: &str) -> Result<ModelInfo> {
    let c_path = CString::new(model_path)?;

    // SAFETY: c_path is a valid null-terminated string.
    let mut info = unsafe { talu_sys::talu_describe(c_path.as_ptr()) };

    // Check for error
    if !info.error_msg.is_null() {
        // SAFETY: error_msg is a valid C string from the C API.
        let msg = unsafe { CStr::from_ptr(info.error_msg) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: info must be freed.
        unsafe { talu_sys::talu_model_info_free(&mut info) };
        return Err(crate::error::Error::generic(msg));
    }

    let model_type = if info.model_type.is_null() {
        String::new()
    } else {
        // SAFETY: model_type is a valid C string from the C API.
        unsafe { CStr::from_ptr(info.model_type) }
            .to_string_lossy()
            .into_owned()
    };

    let architecture = if info.architecture.is_null() {
        String::new()
    } else {
        // SAFETY: architecture is a valid C string from the C API.
        unsafe { CStr::from_ptr(info.architecture) }
            .to_string_lossy()
            .into_owned()
    };

    let result = ModelInfo {
        vocab_size: info.vocab_size,
        hidden_size: info.hidden_size,
        num_layers: info.num_layers,
        num_heads: info.num_heads,
        num_kv_heads: info.num_kv_heads,
        intermediate_size: info.intermediate_size,
        max_seq_len: info.max_seq_len,
        head_dim: info.head_dim,
        rope_theta: info.rope_theta,
        norm_eps: info.norm_eps,
        quant_method: QuantMethod::from(info.quant_method),
        quant_bits: info.quant_bits,
        quant_group_size: info.quant_group_size,
        model_type,
        architecture,
        tie_word_embeddings: info.tie_word_embeddings,
        use_gelu: info.use_gelu,
        num_experts: info.num_experts,
        experts_per_token: info.experts_per_token,
    };

    // SAFETY: info must be freed after use.
    unsafe { talu_sys::talu_model_info_free(&mut info) };

    Ok(result)
}

/// Gets the generation config from a model directory.
pub fn get_generation_config(model_dir: &str) -> Result<GenerationConfigInfo> {
    let c_path = CString::new(model_dir)?;
    let mut info = talu_sys::GenerationConfigInfo::default();

    // SAFETY: c_path is valid, info is a valid mutable pointer.
    let rc = unsafe { talu_sys::talu_get_generation_config(c_path.as_ptr(), &mut info) };

    if rc != 0 {
        return Err(error_from_last_or("Failed to get generation config"));
    }

    Ok(GenerationConfigInfo {
        temperature: info.temperature,
        top_k: info.top_k,
        top_p: info.top_p,
        do_sample: info.do_sample,
    })
}

/// Gets the EOS (end-of-sequence) tokens for a model.
pub fn get_eos_tokens(model_dir: &str) -> Vec<u32> {
    let Ok(c_path) = CString::new(model_dir) else {
        return Vec::new();
    };

    // SAFETY: c_path is a valid null-terminated string.
    let result = unsafe { talu_sys::talu_get_eos_tokens(c_path.as_ptr()) };

    if result.tokens.is_null() || result.num_tokens == 0 {
        return Vec::new();
    }

    // SAFETY: tokens is a valid pointer with num_tokens elements.
    let tokens = unsafe { std::slice::from_raw_parts(result.tokens, result.num_tokens) };
    let list = tokens.to_vec();

    // SAFETY: tokens was allocated by talu and must be freed.
    unsafe { talu_sys::talu_tokens_free(result.tokens, result.num_tokens) };

    list
}

/// Execution plan showing which kernels will be used for a model.
///
/// This provides static analysis of kernel selection based on model config.
/// Use this to understand which code paths need optimization for a given model.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Matmul kernel that will be used (e.g., "matmul_bf16", "matmul_grouped_affine_u4")
    pub matmul_kernel: String,
    /// Attention implementation (e.g., "MultiHeadAttention", "GroupedQueryAttention")
    pub attention_type: String,
    /// FFN type (e.g., "SwiGLU(SiLU)", "MoE(SiLU)")
    pub ffn_type: String,
    /// Number of transformer layers
    pub num_layers: i32,
    /// Hidden dimension
    pub hidden_size: i32,
    /// Number of attention heads
    pub num_heads: i32,
    /// Number of key-value heads
    pub num_kv_heads: i32,
    /// Head dimension
    pub head_dim: i32,
    /// Number of MoE experts (0 if not MoE)
    pub num_experts: i32,
    /// Experts per token for MoE
    pub experts_per_token: i32,
    /// Quantization bits
    pub quant_bits: i32,
    /// Quantization group size
    pub quant_group_size: i32,
    /// Whether model uses grouped-query attention
    pub uses_gqa: bool,
    /// Whether model uses mixture of experts
    pub uses_moe: bool,
    /// Whether model is quantized
    pub uses_quantization: bool,
    /// Whether model uses GELU activation
    pub uses_gelu: bool,
    /// Whether model type is supported by talu's inference engine
    pub is_supported: bool,
}

impl ExecutionPlan {
    /// Print a detailed execution plan to stdout.
    pub fn print_plan(&self) {
        println!("{}", "=".repeat(64));
        println!("EXECUTION PLAN");
        println!("{}", "=".repeat(64));
        println!();
        println!("ARCHITECTURE");
        println!("  Layers:         {}", self.num_layers);
        println!("  Hidden Size:    {}", self.hidden_size);
        println!("  Attention Heads:{}", self.num_heads);
        println!("  KV Heads:       {}", self.num_kv_heads);
        println!("  Head Dimension: {}", self.head_dim);
        println!();
        println!("KERNEL SELECTION");
        println!("  Matmul:         {}", self.matmul_kernel);
        println!("  Attention:      {}", self.attention_type);
        println!("  FFN:            {}", self.ffn_type);
        if self.uses_moe {
            println!();
            println!("MIXTURE OF EXPERTS");
            println!("  Num Experts:    {}", self.num_experts);
            println!("  Top-K:          {}", self.experts_per_token);
        }
        if self.uses_quantization {
            println!();
            println!("QUANTIZATION");
            println!("  Bits:           {}", self.quant_bits);
            println!("  Group Size:     {}", self.quant_group_size);
        }
        println!();
        println!("CODE PATHS TO OPTIMIZE");
        println!(
            "  • compute/ops/matmul_primitives.zig → {}",
            self.matmul_kernel
        );
        println!(
            "  • inference/backend/cpu/kernels/attention.zig → {}",
            self.attention_type
        );
        if self.uses_moe {
            println!(
                "  • inference/backend/cpu/kernels/moe.zig → {}",
                self.ffn_type
            );
        } else {
            println!(
                "  • inference/backend/cpu/kernels/ffn.zig → {}",
                self.ffn_type
            );
        }
        println!("  • inference/backend/cpu/kernels/norm.zig → rmsnormForward");
        println!("{}", "=".repeat(64));
    }
}

/// Gets the execution plan for a model, showing which kernels will be used.
///
/// This is static analysis based on config.json - no model loading required.
/// Use this to understand which code paths to optimize for a given model.
pub fn execution_plan(model_path: &str) -> Result<ExecutionPlan> {
    let c_path = CString::new(model_path)?;

    // First get the model info
    // SAFETY: c_path is a valid null-terminated string.
    let mut info = unsafe { talu_sys::talu_describe(c_path.as_ptr()) };

    // Check for error
    if !info.error_msg.is_null() {
        // SAFETY: error_msg is a valid C string from the C API.
        let msg = unsafe { CStr::from_ptr(info.error_msg) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: info must be freed.
        unsafe { talu_sys::talu_model_info_free(&mut info) };
        return Err(crate::error::Error::generic(msg));
    }

    // Get execution plan from model info
    // SAFETY: info is a valid ModelInfo from talu_describe.
    let plan_info = unsafe { talu_sys::talu_execution_plan(&mut info) };

    // Free the model info
    // SAFETY: info must be freed after use.
    unsafe { talu_sys::talu_model_info_free(&mut info) };

    // Check for error in plan
    if !plan_info.error_msg.is_null() {
        // SAFETY: error_msg is a valid C string from the C API.
        let msg = unsafe { CStr::from_ptr(plan_info.error_msg) }
            .to_string_lossy()
            .into_owned();
        return Err(crate::error::Error::generic(msg));
    }

    // Extract strings
    let matmul_kernel = if plan_info.matmul_kernel.is_null() {
        String::new()
    } else {
        // SAFETY: matmul_kernel is a valid C string from the C API.
        unsafe { CStr::from_ptr(plan_info.matmul_kernel) }
            .to_string_lossy()
            .into_owned()
    };

    let attention_type = if plan_info.attention_type.is_null() {
        String::new()
    } else {
        // SAFETY: attention_type is a valid C string from the C API.
        unsafe { CStr::from_ptr(plan_info.attention_type) }
            .to_string_lossy()
            .into_owned()
    };

    let ffn_type = if plan_info.ffn_type.is_null() {
        String::new()
    } else {
        // SAFETY: ffn_type is a valid C string from the C API.
        unsafe { CStr::from_ptr(plan_info.ffn_type) }
            .to_string_lossy()
            .into_owned()
    };

    Ok(ExecutionPlan {
        matmul_kernel,
        attention_type,
        ffn_type,
        num_layers: plan_info.num_layers,
        hidden_size: plan_info.hidden_size,
        num_heads: plan_info.num_heads,
        num_kv_heads: plan_info.num_kv_heads,
        head_dim: plan_info.head_dim,
        num_experts: plan_info.num_experts,
        experts_per_token: plan_info.experts_per_token,
        quant_bits: plan_info.quant_bits,
        quant_group_size: plan_info.quant_group_size,
        uses_gqa: plan_info.uses_gqa,
        uses_moe: plan_info.uses_moe,
        uses_quantization: plan_info.uses_quantization,
        uses_gelu: plan_info.uses_gelu,
        is_supported: plan_info.is_supported,
    })
}

/// Applies a chat template to messages.
pub fn apply_chat_template(
    model_path: &str,
    messages_json: &str,
    add_generation_prompt: bool,
) -> Result<String> {
    let c_model = CString::new(model_path)?;
    let c_messages = CString::new(messages_json)?;
    let mut out: *mut c_char = std::ptr::null_mut();

    // SAFETY: All pointers are valid.
    let rc = unsafe {
        talu_sys::talu_apply_chat_template(
            c_model.as_ptr(),
            c_messages.as_ptr(),
            if add_generation_prompt { 1 } else { 0 },
            &mut out as *mut _ as *mut c_void,
        )
    };

    if rc != 0 || out.is_null() {
        return Err(error_from_last_or("Failed to apply chat template"));
    }

    // SAFETY: out is a valid C string from the C API.
    let result = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .into_owned();

    // SAFETY: out was allocated by talu and must be freed.
    unsafe { talu_sys::talu_text_free(out) };

    Ok(result)
}
