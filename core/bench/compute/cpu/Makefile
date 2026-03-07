ROOT_DIR := $(abspath $(CURDIR)/../../../..)

model ?=
MODEL ?= $(model)
PROFILE ?= bw
WARMUP ?= 8
ITERS ?= 24
FORMAT ?= table

.PHONY: default help run-model add mul rms rms_bf16 rms_f16 smx scv ssm delta gdnorm gate gate_long mmthr mm_bf16 mm_f16 mm_u4 mm_u8 micro all quick steady qwen35 role_attn_q role_attn_k role_attn_v role_attn_out role_ffn_gate role_ffn_down

default:
ifeq ($(strip $(MODEL)),)
	@$(MAKE) help
else
	@$(MAKE) run-model MODEL="$(MODEL)" PROFILE="$(PROFILE)" FORMAT="$(FORMAT)" WARMUP="$(WARMUP)" ITERS="$(ITERS)"
endif

help:
	@echo "CPU compute benchmark helpers"
	@echo ""
	@echo "Targets:"
	@echo "  make -C core/bench/compute/cpu model=qwen3_5"
	@echo "  make -C core/bench/compute/cpu add"
	@echo "  make -C core/bench/compute/cpu mul"
	@echo "  make -C core/bench/compute/cpu rms"
	@echo "  make -C core/bench/compute/cpu rms_bf16"
	@echo "  make -C core/bench/compute/cpu rms_f16"
	@echo "  make -C core/bench/compute/cpu smx"
	@echo "  make -C core/bench/compute/cpu scv"
	@echo "  make -C core/bench/compute/cpu ssm"
	@echo "  make -C core/bench/compute/cpu delta"
	@echo "  make -C core/bench/compute/cpu gdnorm"
	@echo "  make -C core/bench/compute/cpu gate"
	@echo "  make -C core/bench/compute/cpu gate_long"
	@echo "  make -C core/bench/compute/cpu rope"
	@echo "  make -C core/bench/compute/cpu rope_f16"
	@echo "  make -C core/bench/compute/cpu rope_bf16"
	@echo "  make -C core/bench/compute/cpu sdpa"
	@echo "  make -C core/bench/compute/cpu sdpa_f16"
	@echo "  make -C core/bench/compute/cpu sdpa_bf16"
	@echo "  make -C core/bench/compute/cpu mmthr"
	@echo "  make -C core/bench/compute/cpu mm_bf16"
	@echo "  make -C core/bench/compute/cpu mm_f16"
	@echo "  make -C core/bench/compute/cpu mm_u4"
	@echo "  make -C core/bench/compute/cpu mm_u8"
	@echo "  make -C core/bench/compute/cpu decode_bf16"
	@echo "  make -C core/bench/compute/cpu decode_u4"
	@echo "  make -C core/bench/compute/cpu decode_u8"
	@echo "  make -C core/bench/compute/cpu qwen35"
	@echo "  make -C core/bench/compute/cpu micro"
	@echo "  make -C core/bench/compute/cpu all"
	@echo "  make -C core/bench/compute/cpu quick"
	@echo "  make -C core/bench/compute/cpu steady"
	@echo ""
	@echo "Overrides:"
	@echo "  PROFILE=ci|bw FORMAT=table|csv|tsv WARMUP=<N> ITERS=<N>"

run-model:
	@cd "$(ROOT_DIR)" && \
	zig build bench-cpu-compute -Drelease -- \
		--preset "$(MODEL)" \
		--scenario all \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

add mul rms rms_bf16 rms_f16 smx scv ssm delta gdnorm gate gate_long rope rope_f16 rope_bf16 sdpa sdpa_f16 sdpa_bf16 mmthr mm_bf16 mm_f16 mm_u4 mm_u8 decode_bf16 decode_u4 decode_u8 micro all role_attn_q role_attn_k role_attn_v role_attn_out role_ffn_gate role_ffn_down:
	@cd "$(ROOT_DIR)" && \
	zig build bench-cpu-compute -Drelease -- \
		--scenario "$@" \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

qwen35:
	@$(MAKE) run-model MODEL=qwen3_5 PROFILE="$(PROFILE)" FORMAT="$(FORMAT)" WARMUP="$(WARMUP)" ITERS="$(ITERS)"

quick:
	@$(MAKE) all PROFILE=ci FORMAT=table WARMUP=2 ITERS=5

steady:
	@$(MAKE) all PROFILE=bw FORMAT=table WARMUP=4 ITERS=9
