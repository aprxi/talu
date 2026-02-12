//! Integration tests for `/v1/conversations` endpoints.
//!
//! Note: Search and tag filtering tests are now in `server/search/` module,
//! which tests the `POST /v1/search` endpoint.

mod batch;
mod delete;
mod fork;
mod get;
mod list;
mod patch;
mod routing;

use std::ffi::CString;
use std::path::Path;
use talu::ChatHandle;

/// Seed a test session in TaluDB.
///
/// Creates a session with metadata and a user message so that the
/// conversation has at least one item. Returns the session_id.
pub fn seed_session(db_path: &Path, session_id: &str, title: &str, model: &str) -> String {
    let db_str = db_path.to_str().expect("db_path to str");
    let chat = ChatHandle::new(None).expect("create chat");
    chat.set_storage_db(db_str, session_id)
        .expect("set storage db");
    chat.notify_session_update(Some(model), Some(title), Some("active"))
        .expect("notify session update");
    chat.append_user_message("Hello, world!")
        .expect("append user message");
    // Drop flushes to storage.
    drop(chat);
    session_id.to_string()
}

/// Seed a session with multiple user messages.
///
/// Creates a session with `n` user messages (item IDs 0..n-1).
pub fn seed_session_with_messages(
    db_path: &Path,
    session_id: &str,
    title: &str,
    model: &str,
    messages: &[&str],
) -> String {
    let db_str = db_path.to_str().expect("db_path to str");
    let chat = ChatHandle::new(None).expect("create chat");
    chat.set_storage_db(db_str, session_id)
        .expect("set storage db");
    chat.notify_session_update(Some(model), Some(title), Some("active"))
        .expect("notify session update");
    for msg in messages {
        chat.append_user_message(msg).expect("append user message");
    }
    drop(chat);
    session_id.to_string()
}

/// Seed a session with a system_prompt set.
///
/// Calls the C API directly since `ChatHandle::notify_session_update`
/// doesn't expose the `system_prompt` parameter.
pub fn seed_session_with_system_prompt(
    db_path: &Path,
    session_id: &str,
    title: &str,
    model: &str,
    system_prompt: &str,
) -> String {
    let db_str = db_path.to_str().expect("db_path to str");
    let chat = ChatHandle::new(None).expect("create chat");
    chat.set_storage_db(db_str, session_id)
        .expect("set storage db");

    let c_model = CString::new(model).expect("model cstr");
    let c_title = CString::new(title).expect("title cstr");
    let c_marker = CString::new("active").expect("marker cstr");
    let c_prompt = CString::new(system_prompt).expect("system_prompt cstr");
    let rc = unsafe {
        talu_sys::talu_chat_notify_session_update(
            chat.as_ptr(),
            c_model.as_ptr(),
            c_title.as_ptr(),
            c_prompt.as_ptr(),
            std::ptr::null(), // config_json
            c_marker.as_ptr(),
            std::ptr::null(), // parent_session_id
            std::ptr::null(), // group_id
            std::ptr::null(), // metadata_json
            std::ptr::null(), // source_doc_id
        )
    };
    assert_eq!(rc, 0, "notify_session_update with system_prompt failed");
    chat.append_user_message("Hello, world!")
        .expect("append user message");
    drop(chat);
    session_id.to_string()
}

/// Seed a session with a group_id set.
///
/// Calls the C API directly since `ChatHandle::notify_session_update`
/// doesn't expose the `group_id` parameter.
pub fn seed_session_with_group(
    db_path: &Path,
    session_id: &str,
    title: &str,
    model: &str,
    group_id: &str,
) -> String {
    let db_str = db_path.to_str().expect("db_path to str");
    let chat = ChatHandle::new(None).expect("create chat");
    chat.set_storage_db(db_str, session_id)
        .expect("set storage db");

    // Call C API directly to pass group_id
    let c_model = CString::new(model).expect("model cstr");
    let c_title = CString::new(title).expect("title cstr");
    let c_status = CString::new("active").expect("status cstr");
    let c_group = CString::new(group_id).expect("group cstr");
    let rc = unsafe {
        talu_sys::talu_chat_notify_session_update(
            chat.as_ptr(),
            c_model.as_ptr(),
            c_title.as_ptr(),
            std::ptr::null(), // system_prompt
            std::ptr::null(), // config_json
            c_status.as_ptr(),
            std::ptr::null(), // parent_session_id
            c_group.as_ptr(),
            std::ptr::null(), // metadata_json
            std::ptr::null(), // source_doc_id
        )
    };
    assert_eq!(rc, 0, "notify_session_update with group_id failed");
    chat.append_user_message("Hello, world!")
        .expect("append user message");
    drop(chat);
    session_id.to_string()
}

/// Create a ServerConfig with bucket set.
pub fn conversation_config(bucket: &Path) -> crate::server::common::ServerConfig {
    let mut config = crate::server::common::ServerConfig::new();
    config.bucket = Some(bucket.to_path_buf());
    config
}

/// Create a ServerConfig with --no-bucket (storage disabled).
pub fn no_bucket_config() -> crate::server::common::ServerConfig {
    let mut config = crate::server::common::ServerConfig::new();
    config.no_bucket = true;
    config
}

/// Seed a session with tags.
///
/// Creates the session, then creates tag records and associates them with the session.
/// This ensures tags are properly stored in both the tags table and conversation_tags table,
/// allowing resolve_tags_for_session to return them.
pub fn seed_session_with_tags(
    db_path: &Path,
    session_id: &str,
    title: &str,
    model: &str,
    tags: &[&str],
) -> String {
    let db_str = db_path.to_str().expect("db_path to str");
    let chat = ChatHandle::new(None).expect("create chat");
    chat.set_storage_db(db_str, session_id)
        .expect("set storage db");

    // Build metadata_json with tags (for FTS filtering)
    let tags_json: Vec<String> = tags.iter().map(|t| format!("\"{}\"", t)).collect();
    let metadata_json = format!("{{\"tags\":[{}]}}", tags_json.join(","));

    let c_model = CString::new(model).expect("model cstr");
    let c_title = CString::new(title).expect("title cstr");
    let c_marker = CString::new("active").expect("marker cstr");
    let c_metadata = CString::new(metadata_json).expect("metadata cstr");
    let rc = unsafe {
        talu_sys::talu_chat_notify_session_update(
            chat.as_ptr(),
            c_model.as_ptr(),
            c_title.as_ptr(),
            std::ptr::null(), // system_prompt
            std::ptr::null(), // config_json
            c_marker.as_ptr(),
            std::ptr::null(), // parent_session_id
            std::ptr::null(), // group_id
            c_metadata.as_ptr(),
            std::ptr::null(), // source_doc_id
        )
    };
    assert_eq!(rc, 0, "notify_session_update with tags failed");
    chat.append_user_message("Hello, world!")
        .expect("append user message");
    drop(chat);

    // Create tags and associate them with the session
    let c_db_path = CString::new(db_str).expect("db_path cstr");
    let c_session_id = CString::new(session_id).expect("session_id cstr");
    for tag_name in tags {
        // Generate a tag_id from the name (use name as id for simplicity)
        let tag_id = format!("tag-{}", tag_name);
        let c_tag_id = CString::new(tag_id.as_str()).expect("tag_id cstr");
        let c_tag_name = CString::new(*tag_name).expect("tag_name cstr");

        // Create the tag
        let rc = unsafe {
            talu_sys::talu_storage_create_tag(
                c_db_path.as_ptr(),
                c_tag_id.as_ptr(),
                c_tag_name.as_ptr(),
                std::ptr::null(), // color
                std::ptr::null(), // description
                std::ptr::null(), // group_id
            )
        };
        assert_eq!(rc, 0, "talu_storage_create_tag failed for {}", tag_name);

        // Associate the tag with the session
        let rc = unsafe {
            talu_sys::talu_storage_add_conversation_tag(
                c_db_path.as_ptr(),
                c_session_id.as_ptr(),
                c_tag_id.as_ptr(),
            )
        };
        assert_eq!(
            rc, 0,
            "talu_storage_add_conversation_tag failed for {}",
            tag_name
        );
    }

    session_id.to_string()
}
