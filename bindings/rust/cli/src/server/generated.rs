#[allow(clippy::all, dead_code)]
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#[allow(unused_imports, unused_variables)]
pub mod openapi {
    include!(concat!(env!("OUT_DIR"), "/openapi.rs"));
}

pub use openapi::*;
