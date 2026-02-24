//! Safe Rust wrappers for the Responses API (Item-Based Architecture).
//!
//! This module provides safe access to conversation items, enabling inspection
//! of messages, function calls, reasoning content, and more.
//!
//! # Ownership Model
//!
//! There are two handle types:
//! - [`ResponsesHandle`]: Owns the conversation and frees it on drop.
//! - [`ResponsesRef`]: Borrows a conversation owned by something else (e.g., a `ChatHandle`).
//!
//! Both types provide read access via the [`ResponsesView`] trait.

use crate::error::error_from_last_or;
use crate::Result;
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::os::raw::{c_char, c_void};
use talu_sys;

// Re-export enums from talu_sys for convenience
pub use talu_sys::{ContentType, ImageDetail, ItemStatus, ItemType, MessageRole};

/// Returns the OpenResponses SSE delta event name for a streamed token.
pub fn stream_delta_event_name(item_type: ItemType, content_type: ContentType) -> &'static str {
    match content_type {
        ContentType::ReasoningText => "response.reasoning.delta",
        ContentType::SummaryText => "response.reasoning_summary_text.delta",
        ContentType::Refusal => "response.refusal.delta",
        _ if item_type == ItemType::FunctionCall => "response.function_call_arguments.delta",
        _ => "response.output_text.delta",
    }
}

/// Returns the OpenResponses SSE done event name for a completed content part.
pub fn stream_done_event_name(item_type: ItemType, content_type: ContentType) -> &'static str {
    match content_type {
        ContentType::ReasoningText => "response.reasoning.done",
        ContentType::SummaryText => "response.reasoning_summary_text.done",
        ContentType::Refusal => "response.refusal.done",
        _ if item_type == ItemType::FunctionCall => "response.function_call_arguments.done",
        _ => "response.output_text.done",
    }
}

/// Maps a content type to the OpenResponses content part discriminator.
pub fn stream_content_part_type(content_type: ContentType) -> &'static str {
    match content_type {
        ContentType::ReasoningText => "reasoning_text",
        ContentType::OutputText => "output_text",
        ContentType::Refusal => "refusal",
        ContentType::SummaryText => "summary_text",
        _ => "output_text",
    }
}

// =============================================================================
// Trait for shared read-only operations
// =============================================================================

/// Generate a new session ID.
pub fn new_session_id() -> Result<String> {
    let mut out: *const c_char = std::ptr::null();
    // SAFETY: out is a valid out pointer.
    let rc = unsafe { talu_sys::talu_session_id_new(&mut out as *mut _ as *mut std::ffi::c_void) };
    if rc != 0 || out.is_null() {
        return Err(error_from_last_or("Failed to generate session ID"));
    }
    // SAFETY: out is a valid C string returned by the C API.
    let session_id = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: out was allocated by talu and must be freed.
    unsafe { talu_sys::talu_text_free(out) };
    Ok(session_id)
}

// SAFETY (module-wide invariant): All `ResponsesView` trait methods and
// `ResponsesHandle` mutation methods call C API functions through
// `self.as_ptr()`, which is guaranteed to be a valid, non-null handle by
// construction (checked at creation time in `new()` / `from_raw_owned()` /
// `from_raw()`).  The C API returns either:
//   - scalar values (counts, type codes, status codes) — no pointer concerns;
//   - C string pointers (`*const c_char`) that are valid for the lifetime of
//     the handle — safe to wrap with `CStr::from_ptr` while the handle lives;
//   - `(ptr, len)` pairs for byte slices — safe to wrap with
//     `slice::from_raw_parts` when ptr is non-null and len > 0.
// Every unsafe block below relies on these invariants.  Null-pointer guards
// are applied before every `CStr::from_ptr` / `slice::from_raw_parts` call.

/// Trait for types that provide read access to a conversation's items.
///
/// Both owned [`ResponsesHandle`] and borrowed [`ResponsesRef`] implement this trait.
pub trait ResponsesView {
    /// Returns the raw pointer to the underlying handle.
    /// Note: The C API uses `*mut` for all handle parameters, even for read-only operations.
    fn as_ptr(&self) -> *mut talu_sys::ResponsesHandle;

    /// Returns the number of items in the conversation.
    fn item_count(&self) -> usize {
        // SAFETY: self.as_ptr() is a valid handle (module invariant).
        unsafe { talu_sys::talu_responses_item_count(self.as_ptr()) }
    }

    /// Returns the type of item at the given index.
    fn item_type(&self, index: usize) -> ItemType {
        // SAFETY: self.as_ptr() is a valid handle (module invariant).
        let type_code = unsafe { talu_sys::talu_responses_item_type(self.as_ptr(), index) };
        ItemType::from(type_code)
    }

    /// Gets the item header at the given index.
    fn get_item(&self, index: usize) -> Result<Item> {
        let mut c_item = talu_sys::CItem::default();
        // SAFETY: self.as_ptr() is a valid handle; c_item is a valid out-param (module invariant).
        let rc = unsafe { talu_sys::talu_responses_get_item(self.as_ptr(), index, &mut c_item) };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Failed to get item at index {}",
                index
            )));
        }

        // Fetch generation_json separately
        let generation = self.get_item_generation_json(index);

        Ok(item_from_c(&c_item, generation))
    }

    /// Gets the generation JSON for an item (if present).
    fn get_item_generation_json(&self, index: usize) -> Option<String> {
        let mut ptr: *const u8 = std::ptr::null();
        let mut len: usize = 0;
        // SAFETY: self.as_ptr() is a valid handle; ptr/len are valid out-params.
        let rc = unsafe {
            talu_sys::talu_responses_item_get_generation_json(
                self.as_ptr(),
                index,
                std::ptr::addr_of_mut!(ptr) as *mut std::ffi::c_void,
                std::ptr::addr_of_mut!(len) as *mut std::ffi::c_void,
            )
        };
        if rc != 0 || ptr.is_null() || len == 0 {
            return None;
        }
        // SAFETY: ptr is valid for len bytes (C API contract).
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        Some(String::from_utf8_lossy(slice).into_owned())
    }

    /// Returns concatenated text content for a message item.
    fn message_text(&self, index: usize) -> Result<String> {
        let msg = self.get_message(index)?;
        let mut out = String::new();
        for part_index in 0..msg.content_count {
            let part = self.get_message_content(index, part_index)?;
            if is_text_content(part.content_type) {
                out.push_str(&part.data_utf8_lossy());
            }
        }
        Ok(out)
    }

    /// Returns concatenated text content for a reasoning summary item.
    fn reasoning_summary_text(&self, index: usize) -> Result<String> {
        let reasoning = self.get_reasoning(index)?;
        let mut out = String::new();
        for part_index in 0..reasoning.summary_count {
            let part = self.get_reasoning_summary(index, part_index)?;
            if is_text_content(part.content_type) {
                out.push_str(&part.data_utf8_lossy());
            }
        }
        Ok(out)
    }

    /// Returns the last assistant message text, if any.
    fn last_assistant_message_text(&self) -> Result<Option<String>> {
        let count = self.item_count();
        if count == 0 {
            return Ok(None);
        }
        for index in (0..count).rev() {
            if self.item_type(index) == ItemType::Message {
                let msg = self.get_message(index)?;
                if msg.role == MessageRole::Assistant {
                    return Ok(Some(self.message_text(index)?));
                }
            }
        }
        Ok(None)
    }

    /// Gets message data if the item at index is a message.
    fn get_message(&self, index: usize) -> Result<MessageItem> {
        let mut c_msg = talu_sys::CMessageItem::default();
        // SAFETY: self.as_ptr() is a valid handle; c_msg is a valid out-param (module invariant).
        let rc =
            unsafe { talu_sys::talu_responses_item_as_message(self.as_ptr(), index, &mut c_msg) };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Item at index {} is not a message",
                index
            )));
        }

        let raw_role = if c_msg.raw_role_ptr.is_null() {
            None
        } else {
            // SAFETY: Non-null pointer from C API is a valid null-terminated string
            // owned by the conversation handle (module invariant).
            Some(
                unsafe { CStr::from_ptr(c_msg.raw_role_ptr) }
                    .to_string_lossy()
                    .into_owned(),
            )
        };

        Ok(MessageItem {
            role: MessageRole::from(c_msg.role),
            content_count: c_msg.content_count,
            raw_role,
        })
    }

    /// Gets function call data if the item at index is a function call.
    fn get_function_call(&self, index: usize) -> Result<FunctionCallItem> {
        let mut c_fc = talu_sys::CFunctionCallItem::default();
        // SAFETY: self.as_ptr() is a valid handle; c_fc is a valid out-param (module invariant).
        let rc = unsafe {
            talu_sys::talu_responses_item_as_function_call(self.as_ptr(), index, &mut c_fc)
        };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Item at index {} is not a function call",
                index
            )));
        }

        let name = if c_fc.name_ptr.is_null() {
            String::new()
        } else {
            // SAFETY: Non-null C string pointer from C API (module invariant).
            unsafe { CStr::from_ptr(c_fc.name_ptr) }
                .to_string_lossy()
                .into_owned()
        };

        let call_id = if c_fc.call_id_ptr.is_null() {
            String::new()
        } else {
            // SAFETY: Non-null C string pointer from C API (module invariant).
            unsafe { CStr::from_ptr(c_fc.call_id_ptr) }
                .to_string_lossy()
                .into_owned()
        };

        let arguments = if c_fc.arguments_ptr.is_null() || c_fc.arguments_len == 0 {
            String::new()
        } else {
            // SAFETY: Non-null ptr with valid len from C API (module invariant).
            let slice =
                unsafe { std::slice::from_raw_parts(c_fc.arguments_ptr, c_fc.arguments_len) };
            String::from_utf8_lossy(slice).into_owned()
        };

        Ok(FunctionCallItem {
            name,
            call_id,
            arguments,
        })
    }

    /// Gets function call output data if the item at index is a function call output.
    fn get_function_call_output(&self, index: usize) -> Result<FunctionCallOutputItem> {
        let mut c_fco = talu_sys::CFunctionCallOutputItem::default();
        // SAFETY: self.as_ptr() is a valid handle; c_fco is a valid out-param (module invariant).
        let rc = unsafe {
            talu_sys::talu_responses_item_as_function_call_output(self.as_ptr(), index, &mut c_fco)
        };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Item at index {} is not a function call output",
                index
            )));
        }

        let call_id = if c_fco.call_id_ptr.is_null() {
            String::new()
        } else {
            // SAFETY: Non-null C string pointer from C API (module invariant).
            unsafe { CStr::from_ptr(c_fco.call_id_ptr) }
                .to_string_lossy()
                .into_owned()
        };

        let output_text = if c_fco.is_text_output && !c_fco.output_text_ptr.is_null() {
            // SAFETY: Non-null ptr with valid len from C API (module invariant).
            let slice =
                unsafe { std::slice::from_raw_parts(c_fco.output_text_ptr, c_fco.output_text_len) };
            Some(String::from_utf8_lossy(slice).into_owned())
        } else {
            None
        };

        Ok(FunctionCallOutputItem {
            call_id,
            output_text,
            output_parts_count: c_fco.output_parts_count,
            is_text_output: c_fco.is_text_output,
        })
    }

    /// Gets a content part from a function call output's parts array.
    /// Only valid when is_text_output is false.
    fn get_function_call_output_part(
        &self,
        item_index: usize,
        part_index: usize,
    ) -> Result<ContentPart> {
        let mut c_part = talu_sys::CResponsesContentPart::default();
        // SAFETY: self.as_ptr() is a valid handle; c_part is a valid out-param (module invariant).
        let rc = unsafe {
            talu_sys::talu_responses_item_fco_get_part(
                self.as_ptr(),
                item_index,
                part_index,
                &mut c_part,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Failed to get function call output part at index {}:{}",
                item_index, part_index
            )));
        }
        Ok(content_part_from_c(&c_part))
    }

    /// Gets a message content part without allocation.
    fn get_message_content_ref(
        &self,
        item_index: usize,
        part_index: usize,
    ) -> Result<ContentPartRef<'_>> {
        let mut c_part = talu_sys::CResponsesContentPart::default();
        // SAFETY: self.as_ptr() is a valid handle; c_part is a valid out-param (module invariant).
        let rc = unsafe {
            talu_sys::talu_responses_item_message_get_content(
                self.as_ptr(),
                item_index,
                part_index,
                &mut c_part,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Failed to get message content at index {}:{}",
                item_index, part_index
            )));
        }
        Ok(content_part_ref_from_c(&c_part))
    }

    /// Gets a reasoning content part without allocation.
    fn get_reasoning_content_ref(
        &self,
        item_index: usize,
        part_index: usize,
    ) -> Result<ContentPartRef<'_>> {
        let mut c_part = talu_sys::CResponsesContentPart::default();
        // SAFETY: self.as_ptr() is a valid handle; c_part is a valid out-param (module invariant).
        let rc = unsafe {
            talu_sys::talu_responses_item_reasoning_get_content(
                self.as_ptr(),
                item_index,
                part_index,
                &mut c_part,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Failed to get reasoning content at index {}:{}",
                item_index, part_index
            )));
        }
        Ok(content_part_ref_from_c(&c_part))
    }

    /// Gets a reasoning summary part without allocation.
    fn get_reasoning_summary_ref(
        &self,
        item_index: usize,
        part_index: usize,
    ) -> Result<ContentPartRef<'_>> {
        let mut c_part = talu_sys::CResponsesContentPart::default();
        // SAFETY: self.as_ptr() is a valid handle; c_part is a valid out-param (module invariant).
        let rc = unsafe {
            talu_sys::talu_responses_item_reasoning_get_summary(
                self.as_ptr(),
                item_index,
                part_index,
                &mut c_part,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Failed to get reasoning summary at index {}:{}",
                item_index, part_index
            )));
        }
        Ok(content_part_ref_from_c(&c_part))
    }

    /// Gets a function call output part without allocation.
    fn get_function_call_output_part_ref(
        &self,
        item_index: usize,
        part_index: usize,
    ) -> Result<ContentPartRef<'_>> {
        let mut c_part = talu_sys::CResponsesContentPart::default();
        // SAFETY: self.as_ptr() is a valid handle; c_part is a valid out-param (module invariant).
        let rc = unsafe {
            talu_sys::talu_responses_item_fco_get_part(
                self.as_ptr(),
                item_index,
                part_index,
                &mut c_part,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Failed to get function call output part at index {}:{}",
                item_index, part_index
            )));
        }
        Ok(content_part_ref_from_c(&c_part))
    }

    /// Gets reasoning data if the item at index is a reasoning item.
    fn get_reasoning(&self, index: usize) -> Result<ReasoningItem> {
        let mut c_reasoning = talu_sys::CReasoningItem::default();
        // SAFETY: self.as_ptr() is a valid handle; c_reasoning is a valid out-param (module invariant).
        let rc = unsafe {
            talu_sys::talu_responses_item_as_reasoning(self.as_ptr(), index, &mut c_reasoning)
        };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Item at index {} is not a reasoning item",
                index
            )));
        }

        let encrypted_content = if c_reasoning.encrypted_content_ptr.is_null()
            || c_reasoning.encrypted_content_len == 0
        {
            None
        } else {
            // SAFETY: Non-null ptr with valid len from C API (module invariant).
            let slice = unsafe {
                std::slice::from_raw_parts(
                    c_reasoning.encrypted_content_ptr,
                    c_reasoning.encrypted_content_len,
                )
            };
            Some(slice.to_vec())
        };

        Ok(ReasoningItem {
            content_count: c_reasoning.content_count,
            summary_count: c_reasoning.summary_count,
            encrypted_content,
        })
    }

    /// Gets item reference data if the item at index is an item reference.
    fn get_item_reference(&self, index: usize) -> Result<ItemReferenceItem> {
        let mut c_ref = talu_sys::CItemReferenceItem::default();
        // SAFETY: self.as_ptr() is a valid handle; c_ref is a valid out-param (module invariant).
        let rc = unsafe {
            talu_sys::talu_responses_item_as_item_reference(self.as_ptr(), index, &mut c_ref)
        };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Item at index {} is not an item reference",
                index
            )));
        }

        let id = if c_ref.id_ptr.is_null() {
            String::new()
        } else {
            // SAFETY: Non-null C string pointer from C API (module invariant).
            unsafe { CStr::from_ptr(c_ref.id_ptr) }
                .to_string_lossy()
                .into_owned()
        };

        Ok(ItemReferenceItem { id })
    }

    /// Gets a content part from a message item.
    fn get_message_content(&self, item_index: usize, part_index: usize) -> Result<ContentPart> {
        let mut c_part = talu_sys::CResponsesContentPart::default();
        // SAFETY: self.as_ptr() is a valid handle; c_part is a valid out-param (module invariant).
        let rc = unsafe {
            talu_sys::talu_responses_item_message_get_content(
                self.as_ptr(),
                item_index,
                part_index,
                &mut c_part,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Failed to get content part {} from message {}",
                part_index, item_index
            )));
        }
        Ok(content_part_from_c(&c_part))
    }

    /// Gets the number of content parts in a message.
    fn message_content_count(&self, index: usize) -> usize {
        // SAFETY: self.as_ptr() is a valid handle (module invariant).
        unsafe { talu_sys::talu_responses_item_message_content_count(self.as_ptr(), index) }
    }

    /// Gets a content part from a reasoning item.
    fn get_reasoning_content(&self, item_index: usize, part_index: usize) -> Result<ContentPart> {
        let mut c_part = talu_sys::CResponsesContentPart::default();
        // SAFETY: self.as_ptr() is a valid handle; c_part is a valid out-param (module invariant).
        let rc = unsafe {
            talu_sys::talu_responses_item_reasoning_get_content(
                self.as_ptr(),
                item_index,
                part_index,
                &mut c_part,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Failed to get reasoning content part {} from item {}",
                part_index, item_index
            )));
        }
        Ok(content_part_from_c(&c_part))
    }

    /// Gets the number of content parts in a reasoning item.
    fn reasoning_content_count(&self, index: usize) -> usize {
        // SAFETY: self.as_ptr() is a valid handle (module invariant).
        unsafe { talu_sys::talu_responses_item_reasoning_content_count(self.as_ptr(), index) }
    }

    /// Gets a summary part from a reasoning item.
    fn get_reasoning_summary(&self, item_index: usize, part_index: usize) -> Result<ContentPart> {
        let mut c_part = talu_sys::CResponsesContentPart::default();
        // SAFETY: self.as_ptr() is a valid handle; c_part is a valid out-param (module invariant).
        let rc = unsafe {
            talu_sys::talu_responses_item_reasoning_get_summary(
                self.as_ptr(),
                item_index,
                part_index,
                &mut c_part,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Failed to get reasoning summary part {} from item {}",
                part_index, item_index
            )));
        }
        Ok(content_part_from_c(&c_part))
    }

    /// Gets the number of summary parts in a reasoning item.
    fn reasoning_summary_count(&self, index: usize) -> usize {
        // SAFETY: self.as_ptr() is a valid handle (module invariant).
        unsafe { talu_sys::talu_responses_item_reasoning_summary_count(self.as_ptr(), index) }
    }

    /// Serializes the conversation to Open Responses JSON format.
    ///
    /// `direction`: 0 = request (ItemParam schemas), 1 = response (ItemField schemas)
    fn to_responses_json(&self, direction: u8) -> Result<String> {
        // SAFETY: self.as_ptr() is a valid handle (module invariant).
        let ptr = unsafe { talu_sys::talu_responses_to_responses_json(self.as_ptr(), direction) };
        if ptr.is_null() {
            return Err(error_from_last_or("Failed to serialize to Responses JSON"));
        }
        // SAFETY: Non-null ptr is a heap-allocated C string from the C API.
        let json = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: ptr was allocated by the C API and must be freed exactly once.
        unsafe { talu_sys::talu_text_free(ptr) };
        Ok(json)
    }

    /// Serializes the conversation to legacy Completions JSON format.
    fn to_completions_json(&self) -> Result<String> {
        // SAFETY: self.as_ptr() is a valid handle (module invariant).
        let ptr = unsafe { talu_sys::talu_responses_to_completions_json(self.as_ptr()) };
        if ptr.is_null() {
            return Err(error_from_last_or(
                "Failed to serialize to Completions JSON",
            ));
        }
        // SAFETY: Non-null ptr is a heap-allocated C string from the C API.
        let json = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: ptr was allocated by the C API and must be freed exactly once.
        unsafe { talu_sys::talu_text_free(ptr) };
        Ok(json)
    }
}

// =============================================================================
// ResponsesHandle - Owned conversation
// =============================================================================

/// Owned wrapper for a Responses/Conversation handle.
///
/// Creates and owns a conversation, freeing it on drop.
/// Use [`ResponsesRef`] for borrowed access to conversations owned elsewhere.
pub struct ResponsesHandle {
    ptr: *mut talu_sys::ResponsesHandle,
}

impl ResponsesHandle {
    /// Creates a new empty conversation.
    pub fn new() -> Result<Self> {
        // SAFETY: No preconditions. Returns null on failure.
        let ptr = unsafe { talu_sys::talu_responses_create() };
        if ptr.is_null() {
            return Err(error_from_last_or("Failed to create conversation"));
        }
        Ok(Self { ptr })
    }

    /// Creates a handle from a raw pointer, taking ownership.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid `ResponsesHandle` allocated by the C API.
    /// This struct will free the pointer on Drop.
    pub unsafe fn from_raw_owned(ptr: *mut talu_sys::ResponsesHandle) -> Self {
        Self { ptr }
    }

    /// Creates a new conversation with a session identifier.
    pub fn with_session(session_id: &str) -> Result<Self> {
        let c_session = CString::new(session_id)?;
        // SAFETY: c_session is a valid null-terminated CString.
        let ptr = unsafe { talu_sys::talu_responses_create_with_session(c_session.as_ptr()) };
        if ptr.is_null() {
            return Err(error_from_last_or(
                "Failed to create conversation with session",
            ));
        }
        Ok(Self { ptr })
    }

    /// Returns the raw mutable pointer to the handle.
    pub fn as_mut_ptr(&mut self) -> *mut talu_sys::ResponsesHandle {
        self.ptr
    }

    /// Loads conversation from OpenAI Completions JSON format.
    pub fn load_completions_json(&mut self, json: &str) -> Result<()> {
        let c_json = CString::new(json)?;
        // SAFETY: self.ptr is a valid handle; c_json is a valid CString.
        let rc =
            unsafe { talu_sys::talu_responses_load_completions_json(self.ptr, c_json.as_ptr()) };
        if rc != 0 {
            return Err(error_from_last_or("Failed to load from Completions JSON"));
        }
        Ok(())
    }

    /// Loads items from OpenResponses input format (string or ItemParam array).
    /// Appends to existing items (does not clear).
    pub fn load_responses_json(&mut self, json: &str) -> Result<()> {
        let c_json = CString::new(json)?;
        // SAFETY: self.ptr is a valid handle; c_json is a valid CString.
        let rc = unsafe { talu_sys::talu_responses_load_responses_json(self.ptr, c_json.as_ptr()) };
        if rc != 0 {
            return Err(error_from_last_or("Failed to load from Responses JSON"));
        }
        Ok(())
    }

    /// Clears all items from the conversation.
    pub fn clear(&mut self) {
        // SAFETY: self.ptr is a valid handle.
        unsafe { talu_sys::talu_responses_clear(self.ptr) };
    }

    /// Clears all items except the first system/developer message.
    pub fn clear_keeping_system(&mut self) {
        // SAFETY: self.ptr is a valid handle.
        unsafe { talu_sys::talu_responses_clear_keeping_system(self.ptr) };
    }

    /// Appends a message item to the conversation.
    ///
    /// Returns the index of the new item.
    pub fn append_message(&mut self, role: MessageRole, content: &str) -> Result<usize> {
        // SAFETY: self.ptr is a valid handle; content ptr/len are valid for the str.
        let result = unsafe {
            talu_sys::talu_responses_append_message(
                self.ptr,
                role as u8,
                content.as_ptr(),
                content.len(),
            )
        };
        if result < 0 {
            return Err(error_from_last_or("Failed to append message"));
        }
        Ok(result as usize)
    }

    /// Inserts a message item at the specified index.
    ///
    /// Returns the index of the new item.
    pub fn insert_message(
        &mut self,
        index: usize,
        role: MessageRole,
        content: &str,
    ) -> Result<usize> {
        // SAFETY: self.ptr is a valid handle; content ptr/len are valid for the str.
        let result = unsafe {
            talu_sys::talu_responses_insert_message(
                self.ptr,
                index,
                role as u8,
                content.as_ptr(),
                content.len(),
            )
        };
        if result < 0 {
            return Err(error_from_last_or("Failed to insert message"));
        }
        Ok(result as usize)
    }

    /// Appends a function call item to the conversation.
    ///
    /// Returns the index of the new item.
    pub fn append_function_call(
        &mut self,
        call_id: &str,
        name: &str,
        arguments: &str,
    ) -> Result<usize> {
        let c_call_id = CString::new(call_id)?;
        let c_name = CString::new(name)?;
        // SAFETY: self.ptr is a valid handle; CStrings and arguments ptr/len are valid.
        let result = unsafe {
            talu_sys::talu_responses_append_function_call(
                self.ptr,
                c_call_id.as_ptr(),
                c_name.as_ptr(),
                arguments.as_ptr(),
                arguments.len(),
            )
        };
        if result < 0 {
            return Err(error_from_last_or("Failed to append function call"));
        }
        Ok(result as usize)
    }

    /// Appends a function call output item to the conversation.
    ///
    /// Returns the index of the new item.
    pub fn append_function_call_output(&mut self, call_id: &str, output: &str) -> Result<usize> {
        let c_call_id = CString::new(call_id)?;
        // SAFETY: self.ptr is a valid handle; CString and output ptr/len are valid.
        let result = unsafe {
            talu_sys::talu_responses_append_function_call_output(
                self.ptr,
                c_call_id.as_ptr(),
                output.as_ptr(),
                output.len(),
            )
        };
        if result < 0 {
            return Err(error_from_last_or("Failed to append function call output"));
        }
        Ok(result as usize)
    }

    /// Appends text content to an existing message item.
    pub fn append_text_content(&mut self, item_index: usize, content: &str) -> Result<()> {
        // SAFETY: self.ptr is a valid handle; content ptr/len are valid for the str.
        let rc = unsafe {
            talu_sys::talu_responses_append_text_content(
                self.ptr,
                item_index,
                content.as_ptr(),
                content.len(),
            )
        };
        if rc != 0 {
            return Err(error_from_last_or("Failed to append text content"));
        }
        Ok(())
    }

    /// Removes the last item from the conversation.
    pub fn pop(&mut self) -> Result<()> {
        // SAFETY: self.ptr is a valid handle.
        let rc = unsafe { talu_sys::talu_responses_pop(self.ptr) };
        if rc != 0 {
            return Err(error_from_last_or("Failed to pop item"));
        }
        Ok(())
    }

    /// Removes an item at the specified index.
    pub fn remove(&mut self, index: usize) -> Result<()> {
        // SAFETY: self.ptr is a valid handle.
        let rc = unsafe { talu_sys::talu_responses_remove(self.ptr, index) };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Failed to remove item at index {}",
                index
            )));
        }
        Ok(())
    }

    /// Sets the status of an item at the given index.
    pub fn set_item_status(&mut self, index: usize, status: ItemStatus) -> Result<()> {
        // SAFETY: self.ptr is a valid handle.
        let rc = unsafe { talu_sys::talu_responses_set_item_status(self.ptr, index, status as u8) };
        if rc != 0 {
            return Err(error_from_last_or(&format!(
                "Failed to set item status at index {}",
                index
            )));
        }
        Ok(())
    }

    /// Returns an iterator over all items in the conversation.
    ///
    /// The conversation should not be mutated while iterating.
    pub fn items(&self) -> ItemIterator<'_> {
        ItemIterator {
            ptr: self.ptr,
            index: 0,
            len: self.item_count(),
            _marker: PhantomData,
        }
    }
}

impl ResponsesView for ResponsesHandle {
    fn as_ptr(&self) -> *mut talu_sys::ResponsesHandle {
        self.ptr
    }
}

impl Drop for ResponsesHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: self.ptr was allocated by talu_responses_create and is non-null.
            // Drop is called at most once.
            unsafe { talu_sys::talu_responses_free(self.ptr) };
        }
    }
}

// ResponsesHandle is not Send/Sync by default due to raw pointer
// unsafe impl Send for ResponsesHandle {}
// unsafe impl Sync for ResponsesHandle {}

// =============================================================================
// ResponsesRef - Borrowed conversation (does NOT free on drop)
// =============================================================================

/// Borrowed reference to a conversation owned by something else (e.g., a `ChatHandle`).
///
/// This type provides read-only access to the conversation's items.
/// It does **NOT** free the underlying handle on drop.
///
/// # Safety
///
/// The lifetime `'a` must not outlive the owner of the conversation.
pub struct ResponsesRef<'a> {
    ptr: *const talu_sys::ResponsesHandle,
    _marker: PhantomData<&'a ()>,
}

impl<'a> ResponsesRef<'a> {
    /// Creates a borrowed reference from a raw pointer.
    ///
    /// # Safety
    ///
    /// - The pointer must be a valid ResponsesHandle from the C API.
    /// - The handle must remain valid for the lifetime `'a`.
    /// - The caller must ensure the pointed-to data is not mutated while this reference exists.
    pub unsafe fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr: ptr as *const talu_sys::ResponsesHandle,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over all items in the conversation.
    ///
    /// The underlying conversation must not be mutated while iterating.
    pub fn items(&self) -> ItemIterator<'_> {
        ItemIterator {
            ptr: self.ptr as *mut _,
            index: 0,
            len: self.item_count(),
            _marker: PhantomData,
        }
    }
}

impl<'a> ResponsesView for ResponsesRef<'a> {
    fn as_ptr(&self) -> *mut talu_sys::ResponsesHandle {
        self.ptr as *mut _
    }
}

// ResponsesRef does NOT implement Drop - it's a borrowed reference

// =============================================================================
// Data structures
// =============================================================================

/// Item header (common fields for all item types).
#[derive(Debug, Clone)]
pub struct Item {
    pub id: u64,
    pub item_type: ItemType,
    pub status: ItemStatus,
    pub created_at_ms: i64,
    /// Input token count (prompt tokens for this item).
    pub input_tokens: u32,
    /// Output token count (completion tokens for this item).
    pub output_tokens: u32,
    /// Prefill time in nanoseconds.
    pub prefill_ns: u64,
    /// Generation time in nanoseconds.
    pub generation_ns: u64,
    /// Finish reason (e.g. "stop", "length"). None if not set.
    pub finish_reason: Option<String>,
    /// Generation parameters (model, temperature, etc.) as raw JSON string.
    /// Only populated for assistant messages. None for other roles.
    pub generation: Option<String>,
}

/// Convert a C API `CItem` to the safe `Item` struct.
fn item_from_c(c: &talu_sys::CItem, generation: Option<String>) -> Item {
    let finish_reason = if c.finish_reason_ptr.is_null() {
        None
    } else {
        // SAFETY: Non-null C string pointer from C API (module invariant).
        Some(
            unsafe { CStr::from_ptr(c.finish_reason_ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    };
    Item {
        id: c.id,
        item_type: ItemType::from(c.item_type),
        status: ItemStatus::from(c.status),
        created_at_ms: c.created_at_ms,
        input_tokens: c.input_tokens,
        output_tokens: c.output_tokens,
        prefill_ns: c.prefill_ns,
        generation_ns: c.generation_ns,
        finish_reason,
        generation,
    }
}

/// Message item data.
#[derive(Debug, Clone)]
pub struct MessageItem {
    pub role: MessageRole,
    pub content_count: usize,
    pub raw_role: Option<String>,
}

/// Function call item data.
#[derive(Debug, Clone)]
pub struct FunctionCallItem {
    pub name: String,
    pub call_id: String,
    pub arguments: String,
}

/// Function call output item data.
#[derive(Debug, Clone)]
pub struct FunctionCallOutputItem {
    pub call_id: String,
    pub output_text: Option<String>,
    pub output_parts_count: usize,
    pub is_text_output: bool,
}

/// Reasoning item data.
#[derive(Debug, Clone)]
pub struct ReasoningItem {
    pub content_count: usize,
    pub summary_count: usize,
    pub encrypted_content: Option<Vec<u8>>,
}

/// Item reference data.
#[derive(Debug, Clone)]
pub struct ItemReferenceItem {
    pub id: String,
}

/// Content part data (raw bytes).
#[derive(Debug, Clone)]
pub struct ContentPart {
    pub content_type: ContentType,
    pub image_detail: ImageDetail,
    pub data: Vec<u8>,
    pub secondary_data: Option<Vec<u8>>,
    pub tertiary_data: Option<Vec<u8>>,
    pub quaternary_data: Option<Vec<u8>>,
}

impl ContentPart {
    /// Returns primary data as UTF-8 if valid.
    pub fn data_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.data).ok()
    }

    /// Returns primary data as lossy UTF-8.
    pub fn data_utf8_lossy(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.data)
    }

    /// Returns secondary data as UTF-8 if valid.
    pub fn secondary_str(&self) -> Option<&str> {
        self.secondary_data
            .as_ref()
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
    }

    /// Returns secondary data as lossy UTF-8.
    pub fn secondary_utf8_lossy(&self) -> Option<std::borrow::Cow<'_, str>> {
        self.secondary_data
            .as_ref()
            .map(|bytes| String::from_utf8_lossy(bytes))
    }

    /// Returns tertiary data as UTF-8 if valid.
    pub fn tertiary_str(&self) -> Option<&str> {
        self.tertiary_data
            .as_ref()
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
    }

    /// Returns tertiary data as lossy UTF-8.
    pub fn tertiary_utf8_lossy(&self) -> Option<std::borrow::Cow<'_, str>> {
        self.tertiary_data
            .as_ref()
            .map(|bytes| String::from_utf8_lossy(bytes))
    }

    /// Returns quaternary data as UTF-8 if valid.
    pub fn quaternary_str(&self) -> Option<&str> {
        self.quaternary_data
            .as_ref()
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
    }

    /// Returns quaternary data as lossy UTF-8.
    pub fn quaternary_utf8_lossy(&self) -> Option<std::borrow::Cow<'_, str>> {
        self.quaternary_data
            .as_ref()
            .map(|bytes| String::from_utf8_lossy(bytes))
    }
}

/// Borrowed content part data.
#[derive(Debug, Clone, Copy)]
pub struct ContentPartRef<'a> {
    pub content_type: ContentType,
    pub image_detail: ImageDetail,
    pub data: &'a [u8],
    pub secondary_data: Option<&'a [u8]>,
    pub tertiary_data: Option<&'a [u8]>,
    pub quaternary_data: Option<&'a [u8]>,
}

impl<'a> ContentPartRef<'a> {
    /// Returns primary data as UTF-8 if valid.
    pub fn data_str(&self) -> Option<&'a str> {
        std::str::from_utf8(self.data).ok()
    }

    /// Returns primary data as lossy UTF-8.
    pub fn data_utf8_lossy(&self) -> std::borrow::Cow<'a, str> {
        String::from_utf8_lossy(self.data)
    }

    /// Returns secondary data as UTF-8 if valid.
    pub fn secondary_str(&self) -> Option<&'a str> {
        self.secondary_data
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
    }

    /// Returns secondary data as lossy UTF-8.
    pub fn secondary_utf8_lossy(&self) -> Option<std::borrow::Cow<'a, str>> {
        self.secondary_data
            .map(|bytes| String::from_utf8_lossy(bytes))
    }

    /// Returns tertiary data as UTF-8 if valid.
    pub fn tertiary_str(&self) -> Option<&'a str> {
        self.tertiary_data
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
    }

    /// Returns tertiary data as lossy UTF-8.
    pub fn tertiary_utf8_lossy(&self) -> Option<std::borrow::Cow<'a, str>> {
        self.tertiary_data
            .map(|bytes| String::from_utf8_lossy(bytes))
    }

    /// Returns quaternary data as UTF-8 if valid.
    pub fn quaternary_str(&self) -> Option<&'a str> {
        self.quaternary_data
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
    }

    /// Returns quaternary data as lossy UTF-8.
    pub fn quaternary_utf8_lossy(&self) -> Option<std::borrow::Cow<'a, str>> {
        self.quaternary_data
            .map(|bytes| String::from_utf8_lossy(bytes))
    }
}

// =============================================================================
// Iterator
// =============================================================================

/// Iterator over conversation items.
pub struct ItemIterator<'a> {
    ptr: *mut talu_sys::ResponsesHandle,
    index: usize,
    len: usize,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Iterator for ItemIterator<'a> {
    type Item = Result<Item>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }

        let current_index = self.index;
        let mut c_item = talu_sys::CItem::default();
        // SAFETY: self.ptr is a valid handle; c_item is a valid out-param (module invariant).
        let rc = unsafe { talu_sys::talu_responses_get_item(self.ptr, current_index, &mut c_item) };
        self.index += 1;

        if rc != 0 {
            return Some(Err(error_from_last_or("Failed to get item")));
        }

        // Fetch generation_json separately
        let generation = get_item_generation_json_raw(self.ptr, current_index);

        Some(Ok(item_from_c(&c_item, generation)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len.saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for ItemIterator<'a> {}

// =============================================================================
// Helpers
// =============================================================================

/// Gets generation JSON for an item using a raw handle pointer.
/// Used by ItemIterator which doesn't have access to the trait methods.
fn get_item_generation_json_raw(
    ptr: *mut talu_sys::ResponsesHandle,
    index: usize,
) -> Option<String> {
    let mut data_ptr: *const u8 = std::ptr::null();
    let mut len: usize = 0;
    // SAFETY: ptr is a valid handle; data_ptr/len are valid out-params.
    let rc = unsafe {
        talu_sys::talu_responses_item_get_generation_json(
            ptr,
            index,
            std::ptr::addr_of_mut!(data_ptr) as *mut std::ffi::c_void,
            std::ptr::addr_of_mut!(len) as *mut std::ffi::c_void,
        )
    };
    if rc != 0 || data_ptr.is_null() || len == 0 {
        return None;
    }
    // SAFETY: data_ptr is valid for len bytes (C API contract).
    let slice = unsafe { std::slice::from_raw_parts(data_ptr, len) };
    Some(String::from_utf8_lossy(slice).into_owned())
}

fn is_text_content(content_type: ContentType) -> bool {
    matches!(
        content_type,
        ContentType::InputText
            | ContentType::OutputText
            | ContentType::Text
            | ContentType::ReasoningText
            | ContentType::SummaryText
            | ContentType::Refusal
    )
}

/// Helper to convert C content part to Rust.
fn content_part_from_c(c_part: &talu_sys::CResponsesContentPart) -> ContentPart {
    // SAFETY (all slice::from_raw_parts below): Non-null ptr with valid len
    // from C API, guarded by null/zero checks (module invariant).
    let data = if c_part.data_ptr.is_null() || c_part.data_len == 0 {
        Vec::new()
    } else {
        let slice = unsafe { std::slice::from_raw_parts(c_part.data_ptr, c_part.data_len) };
        slice.to_vec()
    };

    let secondary_data = if c_part.secondary_ptr.is_null() || c_part.secondary_len == 0 {
        None
    } else {
        let slice =
            unsafe { std::slice::from_raw_parts(c_part.secondary_ptr, c_part.secondary_len) };
        Some(slice.to_vec())
    };

    let tertiary_data = if c_part.tertiary_ptr.is_null() || c_part.tertiary_len == 0 {
        None
    } else {
        let slice = unsafe { std::slice::from_raw_parts(c_part.tertiary_ptr, c_part.tertiary_len) };
        Some(slice.to_vec())
    };

    let quaternary_data = if c_part.quaternary_ptr.is_null() || c_part.quaternary_len == 0 {
        None
    } else {
        let slice =
            unsafe { std::slice::from_raw_parts(c_part.quaternary_ptr, c_part.quaternary_len) };
        Some(slice.to_vec())
    };

    ContentPart {
        content_type: ContentType::from(c_part.content_type),
        image_detail: ImageDetail::from(c_part.image_detail),
        data,
        secondary_data,
        tertiary_data,
        quaternary_data,
    }
}

/// Helper to convert C content part to borrowed Rust view.
fn content_part_ref_from_c<'a>(c_part: &talu_sys::CResponsesContentPart) -> ContentPartRef<'a> {
    // SAFETY (all slice::from_raw_parts below): Non-null ptr with valid len
    // from C API, guarded by null/zero checks. Data is owned by the
    // conversation handle and valid for its lifetime (module invariant).
    let data = if c_part.data_ptr.is_null() || c_part.data_len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(c_part.data_ptr, c_part.data_len) }
    };

    let secondary_data = if c_part.secondary_ptr.is_null() || c_part.secondary_len == 0 {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(c_part.secondary_ptr, c_part.secondary_len) })
    };

    let tertiary_data = if c_part.tertiary_ptr.is_null() || c_part.tertiary_len == 0 {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(c_part.tertiary_ptr, c_part.tertiary_len) })
    };

    let quaternary_data = if c_part.quaternary_ptr.is_null() || c_part.quaternary_len == 0 {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(c_part.quaternary_ptr, c_part.quaternary_len) })
    };

    ContentPartRef {
        content_type: ContentType::from(c_part.content_type),
        image_detail: ImageDetail::from(c_part.image_detail),
        data,
        secondary_data,
        tertiary_data,
        quaternary_data,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        stream_content_part_type, stream_delta_event_name, stream_done_event_name, ContentType,
        ItemType,
    };

    #[test]
    fn stream_event_name_maps_function_call() {
        assert_eq!(
            stream_delta_event_name(ItemType::FunctionCall, ContentType::OutputText),
            "response.function_call_arguments.delta"
        );
        assert_eq!(
            stream_done_event_name(ItemType::FunctionCall, ContentType::OutputText),
            "response.function_call_arguments.done"
        );
    }

    #[test]
    fn stream_event_name_maps_reasoning_and_summary() {
        assert_eq!(
            stream_delta_event_name(ItemType::Reasoning, ContentType::ReasoningText),
            "response.reasoning.delta"
        );
        assert_eq!(
            stream_done_event_name(ItemType::Reasoning, ContentType::ReasoningText),
            "response.reasoning.done"
        );
        assert_eq!(
            stream_delta_event_name(ItemType::Reasoning, ContentType::SummaryText),
            "response.reasoning_summary_text.delta"
        );
        assert_eq!(
            stream_done_event_name(ItemType::Reasoning, ContentType::SummaryText),
            "response.reasoning_summary_text.done"
        );
    }

    #[test]
    fn stream_content_part_type_maps_expected_values() {
        assert_eq!(
            stream_content_part_type(ContentType::ReasoningText),
            "reasoning_text"
        );
        assert_eq!(
            stream_content_part_type(ContentType::OutputText),
            "output_text"
        );
        assert_eq!(stream_content_part_type(ContentType::Refusal), "refusal");
        assert_eq!(
            stream_content_part_type(ContentType::SummaryText),
            "summary_text"
        );
    }
}
