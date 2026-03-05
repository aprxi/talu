ROOT_DIR := $(abspath $(CURDIR)/../../..)

PROFILE ?= bw
WARMUP ?= 4
ITERS ?= 20
FORMAT ?= table
THREADS ?=

# Conditional env prefix for thread pinning
ifdef THREADS
  ENV_PREFIX := THREADS=$(THREADS)
else
  ENV_PREFIX :=
endif

# Shared run command
define run_bench
	@cd "$(ROOT_DIR)" && $(ENV_PREFIX) \
	zig build bench-train -Drelease -- \
		--scenario $(1) \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"
endef

.PHONY: help all quick \
        forward_linear forward_attention forward_norm forward_activation \
        forward_rope forward_embedding forward_loss forward_full \
        backward_linear backward_full step_full optimizer_step \
        all-csv all-table

help:
	@echo "Training benchmark helpers"
	@echo ""
	@echo "Targets:"
	@echo "  make -C core/bench/train all              Run all scenarios"
	@echo "  make -C core/bench/train quick            Quick CI run"
	@echo "  make -C core/bench/train forward_linear   Single linear projection"
	@echo "  make -C core/bench/train forward_full     Full forward pass"
	@echo "  make -C core/bench/train backward_full    Full backward pass"
	@echo "  make -C core/bench/train step_full        Full training step"
	@echo ""
	@echo "Overrides:"
	@echo "  PROFILE=ci|bw  FORMAT=table|csv  WARMUP=<N>  ITERS=<N>  THREADS=<N>"

forward_linear:
	$(call run_bench,forward_linear)

forward_attention:
	$(call run_bench,forward_attention)

forward_norm:
	$(call run_bench,forward_norm)

forward_activation:
	$(call run_bench,forward_activation)

forward_rope:
	$(call run_bench,forward_rope)

forward_embedding:
	$(call run_bench,forward_embedding)

forward_loss:
	$(call run_bench,forward_loss)

forward_full:
	$(call run_bench,forward_full)

backward_linear:
	$(call run_bench,backward_linear)

backward_full:
	$(call run_bench,backward_full)

step_full:
	$(call run_bench,step_full)

optimizer_step:
	$(call run_bench,optimizer_step)

all:
	$(call run_bench,all)

quick:
	@$(MAKE) all PROFILE=ci FORMAT=table WARMUP=2 ITERS=3

all-csv:
	@$(MAKE) all FORMAT=csv

all-table:
	@$(MAKE) all FORMAT=table
