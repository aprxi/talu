ROOT_DIR := $(abspath $(CURDIR)/../../../..)

PROFILE ?= bw
WARMUP ?= 8
ITERS ?= 24
FORMAT ?= table

.PHONY: help add mmthr attn ffnq ffnq8 micro decode decd all bw quick add-ci mmthr-ci attn-ci ffnq-ci ffnq8-ci micro-ci decode-ci decd-ci all-ci bw-ci add-table mmthr-table attn-table ffnq-table ffnq8-table micro-table decode-table decd-table all-table bw-table

help:
	@echo "Metal compute benchmark helpers"
	@echo ""
	@echo "Targets:"
	@echo "  make -C core/bench/compute/metal add"
	@echo "  make -C core/bench/compute/metal mmthr"
	@echo "  make -C core/bench/compute/metal attn"
	@echo "  make -C core/bench/compute/metal ffnq"
	@echo "  make -C core/bench/compute/metal ffnq8"
	@echo "  make -C core/bench/compute/metal micro"
	@echo "  make -C core/bench/compute/metal decode"
	@echo "  make -C core/bench/compute/metal decd"
	@echo "  make -C core/bench/compute/metal all"
	@echo "  make -C core/bench/compute/metal bw"
	@echo "  make -C core/bench/compute/metal quick"
	@echo "  make -C core/bench/compute/metal all-ci"
	@echo "  make -C core/bench/compute/metal all-table"
	@echo ""
	@echo "Overrides:"
	@echo "  PROFILE=ci|bw FORMAT=table|csv|tsv WARMUP=<N> ITERS=<N>"

micro:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario micro \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

add:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario add \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

mmthr:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario mmthr \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

attn:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario attn \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

ffnq:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario ffnq_u4 \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

ffnq8:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario ffnq_u8 \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

decode:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario decode \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

decd:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario decd \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

all:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario all \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

bw:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario all \
		--profile bw \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

quick:
	@$(MAKE) add PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) mmthr PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) attn PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) ffnq PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) ffnq8 PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) micro PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) decode PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) decd PROFILE=ci FORMAT=table WARMUP=2 ITERS=5

add-ci:
	@$(MAKE) add FORMAT=csv

mmthr-ci:
	@$(MAKE) mmthr FORMAT=csv

attn-ci:
	@$(MAKE) attn FORMAT=csv

ffnq-ci:
	@$(MAKE) ffnq FORMAT=csv

ffnq8-ci:
	@$(MAKE) ffnq8 FORMAT=csv

micro-ci:
	@$(MAKE) micro FORMAT=csv

decode-ci:
	@$(MAKE) decode FORMAT=csv

decd-ci:
	@$(MAKE) decd FORMAT=csv

all-ci:
	@$(MAKE) all PROFILE=ci FORMAT=csv

bw-ci:
	@$(MAKE) bw FORMAT=csv

add-table:
	@$(MAKE) add FORMAT=table

mmthr-table:
	@$(MAKE) mmthr FORMAT=table

attn-table:
	@$(MAKE) attn FORMAT=table

ffnq-table:
	@$(MAKE) ffnq FORMAT=table

ffnq8-table:
	@$(MAKE) ffnq8 FORMAT=table

micro-table:
	@$(MAKE) micro FORMAT=table

decode-table:
	@$(MAKE) decode FORMAT=table

decd-table:
	@$(MAKE) decd FORMAT=table

all-table:
	@$(MAKE) all FORMAT=table

bw-table:
	@$(MAKE) bw FORMAT=table
