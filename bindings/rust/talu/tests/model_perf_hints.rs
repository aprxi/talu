use talu::model;

#[test]
fn returns_perf_hints_for_registered_architecture() {
    for arch in ["qwen3_5", "qwen3", "lfm2", "gemma3"] {
        let json = model::performance_hints_json(arch)
            .expect("fetch perf hints")
            .expect("registered architecture should have perf hints");
        assert!(json.contains(&format!("\"bench_model\":\"{arch}\"")));
        assert!(json.contains("\"prefill_point_mappings\""));
        assert!(json.contains("\"decode_point_mappings\""));
    }
}

#[test]
fn returns_none_for_unknown_architecture() {
    let json = model::performance_hints_json("definitely_not_real").expect("query unknown hints");
    assert!(json.is_none());
}
