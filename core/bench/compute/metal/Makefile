ROOT_DIR := $(abspath $(CURDIR)/../../../..)

PROFILE ?= bw
WARMUP ?= 8
ITERS ?= 24
FORMAT ?= table

.PHONY: help add mmthr attn scvq ssm ffnq ffnq8 ffnd micro decode decd decq all bw quick add-ci mmthr-ci attn-ci scvq-ci ssm-ci ffnq-ci ffnq8-ci ffnd-ci micro-ci decode-ci decd-ci decq-ci all-ci bw-ci add-table mmthr-table attn-table scvq-table ssm-table ffnq-table ffnq8-table ffnd-table micro-table decode-table decd-table decq-table all-table bw-table

help:
	@echo "Metal compute benchmark helpers"
	@echo ""
	@echo "Targets:"
	@echo "  make -C core/bench/compute/metal add"
	@echo "  make -C core/bench/compute/metal mmthr"
	@echo "  make -C core/bench/compute/metal attn"
	@echo "  make -C core/bench/compute/metal scvq"
	@echo "  make -C core/bench/compute/metal ssm"
	@echo "  make -C core/bench/compute/metal ffnq"
	@echo "  make -C core/bench/compute/metal ffnq8"
	@echo "  make -C core/bench/compute/metal ffnd"
	@echo "  make -C core/bench/compute/metal micro"
	@echo "  make -C core/bench/compute/metal decode"
	@echo "  make -C core/bench/compute/metal decd"
	@echo "  make -C core/bench/compute/metal decq"
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

scvq:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario scvq \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

ssm:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario ssm \
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

ffnd:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario ffnd \
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

decq:
	@cd "$(ROOT_DIR)" && \
	zig build bench-metal-compute -Drelease -- \
		--scenario decq \
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
	@$(MAKE) scvq PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) ssm PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) ffnq PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) ffnq8 PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) ffnd PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) micro PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) decode PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) decd PROFILE=ci FORMAT=table WARMUP=2 ITERS=5
	@$(MAKE) decq PROFILE=ci FORMAT=table WARMUP=2 ITERS=5

add-ci:
	@$(MAKE) add FORMAT=csv

mmthr-ci:
	@$(MAKE) mmthr FORMAT=csv

attn-ci:
	@$(MAKE) attn FORMAT=csv

scvq-ci:
	@$(MAKE) scvq FORMAT=csv

ssm-ci:
	@$(MAKE) ssm FORMAT=csv

ffnq-ci:
	@$(MAKE) ffnq FORMAT=csv

ffnq8-ci:
	@$(MAKE) ffnq8 FORMAT=csv

ffnd-ci:
	@$(MAKE) ffnd FORMAT=csv

micro-ci:
	@$(MAKE) micro FORMAT=csv

decode-ci:
	@$(MAKE) decode FORMAT=csv

decd-ci:
	@$(MAKE) decd FORMAT=csv

decq-ci:
	@$(MAKE) decq FORMAT=csv

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

scvq-table:
	@$(MAKE) scvq FORMAT=table

ssm-table:
	@$(MAKE) ssm FORMAT=table

ffnq-table:
	@$(MAKE) ffnq FORMAT=table

ffnq8-table:
	@$(MAKE) ffnq8 FORMAT=table

ffnd-table:
	@$(MAKE) ffnd FORMAT=table

micro-table:
	@$(MAKE) micro FORMAT=table

decode-table:
	@$(MAKE) decode FORMAT=table

decd-table:
	@$(MAKE) decd FORMAT=table

decq-table:
	@$(MAKE) decq FORMAT=table

all-table:
	@$(MAKE) all FORMAT=table

bw-table:
	@$(MAKE) bw FORMAT=table
