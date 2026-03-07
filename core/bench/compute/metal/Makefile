ROOT_DIR := $(abspath $(CURDIR)/../../../..)

.DEFAULT_GOAL := all

PROFILE ?= bw
WARMUP ?= 8
ITERS ?= 24
FORMAT ?= table

.PHONY: help all all-table

help:
	@echo "Metal compute benchmark helpers"
	@echo ""
	@echo "Targets:"
	@echo "  make -C core/bench/compute/metal all"
	@echo "  make -C core/bench/compute/metal all-table"
	@echo ""
	@echo "Overrides:"
	@echo "  PROFILE=ci|bw FORMAT=table|csv|tsv WARMUP=<N> ITERS=<N>"
	@echo ""
	@echo "For single-scenario runs, use:"
	@echo "  zig build bench-metal-compute -Drelease -- --scenario <name> ..."

all:
	@cd "$(ROOT_DIR)" && \
		zig build bench-metal-compute -Drelease -- \
			--scenario all \
			--profile "$(PROFILE)" \
			--format "$(FORMAT)" \
			--warmup "$(WARMUP)" \
			--iters "$(ITERS)"

all-table:
	@$(MAKE) all FORMAT=table
