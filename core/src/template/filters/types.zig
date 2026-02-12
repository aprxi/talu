//! Common types and utilities for filters
//!
//! Shared imports and type aliases used across all filter modules.

const std = @import("std");
const ast = @import("../ast.zig");
const eval = @import("../eval.zig");

pub const TemplateInput = eval.TemplateInput;
pub const EvalError = eval.EvalError;
pub const Evaluator = eval.Evaluator;
pub const CustomFilter = eval.CustomFilter;
pub const Expr = ast.Expr;

/// Standard filter function signature
pub const FilterFn = *const fn (*Evaluator, TemplateInput, []const *const Expr) EvalError!TemplateInput;
