.PHONY: bench bench-prep bench-info bench-report

MANIFEST     := --manifest-path ../Cargo.toml
BENCH_SCOPES := search storage api rag

# Usage:
#   make bench                           - quick sanity check (<10s)
#   make bench scope=help                - list all scopes and their benchmarks
#   make bench scope=search filter=help  - list benchmarks in "search" scope
#   make bench scope=search              - full Criterion suite for search
#   make bench scope=search filter=batch - only benchmarks matching "batch"
#   make bench-prep dataset=medium       - generate persistent dataset
#   make bench-prep dataset=medium force=1 - recreate from scratch
#   make bench-info                      - show all dataset stats
#   make bench-info dataset=medium       - show stats for one dataset
#   make bench-report                    - all suites with JSON summary

# Extract benchmark names from @bench tags in main.rs (instant, no compilation).
bench-list = grep '// @bench:' $(1)/main.rs | sed 's|.*// @bench: *||'

bench:
ifndef scope
	@echo "bench-quick (sanity check)"
	cargo run $(MANIFEST) --release --example bench_quick
else ifeq ($(scope),help)
	@for s in $(BENCH_SCOPES); do \
		echo "scope=$$s"; \
		$(call bench-list,$$s) | sed 's/^/  /'; \
		echo; \
	done
else ifeq ($(filter),help)
	@echo "scope=$(scope)"
	@$(call bench-list,$(scope)) | sed 's/^/  /'
else
	@echo "criterion: scope=$(scope) filter=$(filter)"
	cargo bench $(MANIFEST) --bench $(scope) $(if $(filter),-- $(filter),)
endif

bench-prep:
ifndef dataset
	@echo "Usage: make bench-prep dataset=small|medium|large|all [force=1]"
else
	cargo run $(MANIFEST) --release --example bench_prep -- --dataset $(dataset) $(if $(force),--force,)
endif

bench-info:
ifndef dataset
	cargo run $(MANIFEST) --release --example bench_prep -- --info all
else
	cargo run $(MANIFEST) --release --example bench_prep -- --info $(dataset)
endif

bench-report:
	cargo criterion $(MANIFEST) --output-format quiet --message-format json 2>/dev/null \
		| python3 ../../../../scripts/bench-summary.py
