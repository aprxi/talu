ROOT_DIR := $(abspath $(CURDIR)/../../..)

TOK ?=
PROFILE ?= bw
WARMUP ?= 4
ITERS ?= 10
FORMAT ?= table
SIZE ?= 1m

.PHONY: help encode_1k encode_100k encode_1m load all quick breakdown \
        normalize_1m pretok_1m bpe_1m bpe_merge_1m bpe_vocab_1m bpe_collect_1m \
        encode_1k-csv encode_100k-csv encode_1m-csv load-csv all-csv \
        encode_1k-table encode_100k-table encode_1m-table load-table all-table

help:
	@echo "Tokenizer benchmark helpers"
	@echo ""
	@echo "Targets:"
	@echo "  make -C core/bench/tokenizer breakdown   TOK=<path> [SIZE=100k]"
	@echo "  make -C core/bench/tokenizer encode_1k   TOK=<path>"
	@echo "  make -C core/bench/tokenizer encode_100k TOK=<path>"
	@echo "  make -C core/bench/tokenizer encode_1m   TOK=<path>"
	@echo "  make -C core/bench/tokenizer all         TOK=<path>"
	@echo "  make -C core/bench/tokenizer quick       TOK=<path>"
	@echo ""
	@echo "Overrides:"
	@echo "  TOK=<path/to/tokenizer.json>  (required)"
	@echo "  SIZE=1k|10k|100k|1m  (for breakdown, default: 1m)"
	@echo "  PROFILE=ci|bw FORMAT=table|csv|tsv WARMUP=<N> ITERS=<N>"

breakdown:
	@test -n "$(TOK)" || { echo "error: TOK=<path/to/tokenizer.json> is required"; exit 1; }
	@cd "$(ROOT_DIR)" && \
	zig build bench-tokenizer -Drelease -- \
		--scenario breakdown \
		--tokenizer "$(TOK)" \
		--size "$(SIZE)" \
		--profile "$(PROFILE)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

encode_1k:
	@test -n "$(TOK)" || { echo "error: TOK=<path/to/tokenizer.json> is required"; exit 1; }
	@cd "$(ROOT_DIR)" && \
	zig build bench-tokenizer -Drelease -- \
		--scenario encode_1k \
		--tokenizer "$(TOK)" \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

encode_100k:
	@test -n "$(TOK)" || { echo "error: TOK=<path/to/tokenizer.json> is required"; exit 1; }
	@cd "$(ROOT_DIR)" && \
	zig build bench-tokenizer -Drelease -- \
		--scenario encode_100k \
		--tokenizer "$(TOK)" \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

encode_1m:
	@test -n "$(TOK)" || { echo "error: TOK=<path/to/tokenizer.json> is required"; exit 1; }
	@cd "$(ROOT_DIR)" && \
	zig build bench-tokenizer -Drelease -- \
		--scenario encode_1m \
		--tokenizer "$(TOK)" \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

load:
	@test -n "$(TOK)" || { echo "error: TOK=<path/to/tokenizer.json> is required"; exit 1; }
	@cd "$(ROOT_DIR)" && \
	zig build bench-tokenizer -Drelease -- \
		--scenario load_json \
		--tokenizer "$(TOK)" \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

all:
	@test -n "$(TOK)" || { echo "error: TOK=<path/to/tokenizer.json> is required"; exit 1; }
	@cd "$(ROOT_DIR)" && \
	zig build bench-tokenizer -Drelease -- \
		--scenario all \
		--tokenizer "$(TOK)" \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

normalize_1m:
	@test -n "$(TOK)" || { echo "error: TOK=<path/to/tokenizer.json> is required"; exit 1; }
	@cd "$(ROOT_DIR)" && \
	zig build bench-tokenizer -Drelease -- \
		--scenario normalize_1m \
		--tokenizer "$(TOK)" \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

pretok_1m:
	@test -n "$(TOK)" || { echo "error: TOK=<path/to/tokenizer.json> is required"; exit 1; }
	@cd "$(ROOT_DIR)" && \
	zig build bench-tokenizer -Drelease -- \
		--scenario pretok_1m \
		--tokenizer "$(TOK)" \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

bpe_1m:
	@test -n "$(TOK)" || { echo "error: TOK=<path/to/tokenizer.json> is required"; exit 1; }
	@cd "$(ROOT_DIR)" && \
	zig build bench-tokenizer -Drelease -- \
		--scenario bpe_1m \
		--tokenizer "$(TOK)" \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

bpe_merge_1m:
	@test -n "$(TOK)" || { echo "error: TOK=<path/to/tokenizer.json> is required"; exit 1; }
	@cd "$(ROOT_DIR)" && \
	zig build bench-tokenizer -Drelease -- \
		--scenario bpe_merge_1m \
		--tokenizer "$(TOK)" \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

bpe_vocab_1m:
	@test -n "$(TOK)" || { echo "error: TOK=<path/to/tokenizer.json> is required"; exit 1; }
	@cd "$(ROOT_DIR)" && \
	zig build bench-tokenizer -Drelease -- \
		--scenario bpe_vocab_1m \
		--tokenizer "$(TOK)" \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

bpe_collect_1m:
	@test -n "$(TOK)" || { echo "error: TOK=<path/to/tokenizer.json> is required"; exit 1; }
	@cd "$(ROOT_DIR)" && \
	zig build bench-tokenizer -Drelease -- \
		--scenario bpe_collect_1m \
		--tokenizer "$(TOK)" \
		--profile "$(PROFILE)" \
		--format "$(FORMAT)" \
		--warmup "$(WARMUP)" \
		--iters "$(ITERS)"

quick:
	@$(MAKE) all PROFILE=ci FORMAT=table WARMUP=2 ITERS=3

encode_1k-csv:
	@$(MAKE) encode_1k FORMAT=csv

encode_100k-csv:
	@$(MAKE) encode_100k FORMAT=csv

encode_1m-csv:
	@$(MAKE) encode_1m FORMAT=csv

load-csv:
	@$(MAKE) load FORMAT=csv

all-csv:
	@$(MAKE) all FORMAT=csv

encode_1k-table:
	@$(MAKE) encode_1k FORMAT=table

encode_100k-table:
	@$(MAKE) encode_100k FORMAT=table

encode_1m-table:
	@$(MAKE) encode_1m FORMAT=table

load-table:
	@$(MAKE) load FORMAT=table

all-table:
	@$(MAKE) all FORMAT=table
