README ?= ../../../../../../README.md
MODEL ?= Qwen/Qwen3.5-2B
TOKENS ?= 120
WARMUP ?= 1
SEED ?= 42
CHAT_TEMPLATE ?= 0
VERBOSE ?= 0
ARTIFACT ?= 0
STEP_MODE ?= multitrace

CXX ?= c++
ZIG ?= zig

MLX_INC := ../../../../../../deps/mlx/include
MLX_LIB_DIR := ../../../../../../deps/mlx/lib
MLX_LIB := $(MLX_LIB_DIR)/libmlx.a
MLX_METALLIB := $(MLX_LIB_DIR)/mlx.metallib
METAL_PATH_ABS := $(abspath $(MLX_METALLIB))
TALU_LIB_DIR := ../../../../../../zig-out/lib
TALU_LIB_DIR_ABS := $(abspath $(TALU_LIB_DIR))

OBJ_DIR := build
BIN_DIR := bin
BRIDGE_OBJ := $(OBJ_DIR)/bridge.o
BRIDGE_CONFIG_TEST_BIN := $(BIN_DIR)/bridge_config_test
CPP_BIN := $(BIN_DIR)/cpp_host
ZIG_BIN := $(BIN_DIR)/scheduled_host
BIN_METALLIB := $(BIN_DIR)/mlx.metallib

CXXFLAGS += -std=c++17 -O3 -I$(MLX_INC) -I. -DMETAL_PATH=\"$(METAL_PATH_ABS)\"
LDFLAGS += $(MLX_LIB) -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate

.PHONY: \
	build build-cpp build-zig \
	prepare-zig-readme prepare-cpp-readme \
	run-zig-readme run-cpp-readme \
	bench bench-readme bench-readme-fresh \
	bench-cpp-readme bench-zig-readme bench-all \
	test-config \
	clean info

build: build-cpp build-zig

$(BRIDGE_OBJ): bridge.cpp bridge.h
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c bridge.cpp -o $(BRIDGE_OBJ)

$(BIN_METALLIB): $(MLX_METALLIB)
	@mkdir -p $(BIN_DIR)
	cp $(MLX_METALLIB) $(BIN_METALLIB)

build-cpp: $(BRIDGE_OBJ) main_cpp.cpp $(BIN_METALLIB)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) main_cpp.cpp $(BRIDGE_OBJ) -o $(CPP_BIN) $(LDFLAGS)

build-zig: $(BRIDGE_OBJ) scheduled.zig bridge.h $(BIN_METALLIB)
	@mkdir -p $(BIN_DIR)
	$(ZIG) build-exe --dep build_options \
		-Mroot=scheduled.zig \
		-Mbuild_options=build_options.zig \
		$(BRIDGE_OBJ) -O ReleaseFast -femit-bin=$(ZIG_BIN) \
		-I. -I$(MLX_INC) $(MLX_LIB) -L$(TALU_LIB_DIR) -ltalu -rpath $(TALU_LIB_DIR_ABS) -lc -lc++ \
		-framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate

$(BRIDGE_CONFIG_TEST_BIN): bridge_config_test.cpp config_parse.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) bridge_config_test.cpp config_parse.cpp -o $(BRIDGE_CONFIG_TEST_BIN)

test-config: $(BRIDGE_CONFIG_TEST_BIN)
	./$(BRIDGE_CONFIG_TEST_BIN)

prepare-cpp-readme: build-cpp

prepare-zig-readme: build-zig

run-cpp-readme:
	@echo "cpp runner is disabled in repository mode."
	@echo "Use: make run-zig-readme MODEL=... TOKENS=..."
	@exit 1

run-zig-readme:
	@echo "============================================================"
	@echo "Model : $(MODEL)"
	@echo "host  : zig (debug/mlx-scheduled scheduler-adapter)"
	@echo "chat template: $(CHAT_TEMPLATE)"
	@echo "verbose: $(VERBOSE)"
	@echo "artifact: $(ARTIFACT)"
	@echo "tokens: $(TOKENS)  warmup: $(WARMUP)  seed: $(SEED)"
	@echo "============================================================"
	@DYLD_LIBRARY_PATH="$(TALU_LIB_DIR_ABS):$$DYLD_LIBRARY_PATH" $(ZIG_BIN) \
		--model $(MODEL) \
		--tokens $(TOKENS) \
		--warmup $(WARMUP) \
		--seed $(SEED) \
		--chat-template $(CHAT_TEMPLATE) \
		--verbose $(VERBOSE) \
		--artifact $(ARTIFACT)

bench-cpp-readme: prepare-cpp-readme run-cpp-readme

bench-zig-readme: prepare-zig-readme run-zig-readme

# stdin-driven benchmark path:
#   head -n50 ../../README.md | make bench MODEL=... TOKENS=...
bench: run-zig-readme

# Benchmark against first 50 lines of $(README), mirroring pymlx behavior.
bench-readme: prepare-zig-readme
	@echo "============================================================"
	@echo "Model : $(MODEL)"
	@echo "host  : zig (debug/mlx-scheduled scheduler-adapter)"
	@echo "chat template: $(CHAT_TEMPLATE)"
	@echo "verbose: $(VERBOSE)"
	@echo "artifact: $(ARTIFACT)"
	@echo "tokens: $(TOKENS)  warmup: $(WARMUP)  seed: $(SEED)"
	@echo "============================================================"
	@head -n50 $(README) | DYLD_LIBRARY_PATH="$(TALU_LIB_DIR_ABS):$$DYLD_LIBRARY_PATH" $(ZIG_BIN) \
		--model $(MODEL) \
		--tokens $(TOKENS) \
		--warmup $(WARMUP) \
		--seed $(SEED) \
		--chat-template $(CHAT_TEMPLATE) \
		--verbose $(VERBOSE) \
		--artifact $(ARTIFACT)

# Full pipeline: rebuild + run.
bench-readme-fresh: clean prepare-zig-readme bench-readme

bench-all: bench-cpp-readme bench-zig-readme

info:
	@echo "model      : $(MODEL)"
	@echo "tokens     : $(TOKENS)"
	@echo "warmup     : $(WARMUP)"
	@echo "seed       : $(SEED)"
	@echo "chat tmpl  : $(CHAT_TEMPLATE)"
	@echo "verbose    : $(VERBOSE)"
	@echo "artifact   : $(ARTIFACT)"
	@echo "mlx lib    : $(MLX_LIB)"
	@echo "metallib   : $(MLX_METALLIB)"

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
