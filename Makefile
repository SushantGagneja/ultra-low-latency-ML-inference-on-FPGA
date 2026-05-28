# VeriTrade-style Makefile for BNN Core Simulation

IVERILOG = iverilog
VVP = vvp
GTKWAVE = gtkwave

IVERILOG_FLAGS = -Wall -g2012 -I rtl -I rtl/testbench

RTL_SRCS = \
	rtl/bram_weights.v \
	rtl/xnor_popcount.v \
	rtl/bnn_core.v \
	rtl/spi_slave.v \
	rtl/bnn_top.v

TB_SRC = rtl/testbench/bnn_core_tb.v

SIM_DIR = sim
SIM_BIN = $(SIM_DIR)/bnn_core.vvp
VCD_FILE = $(SIM_DIR)/bnn_core.vcd

all: run_sim

$(SIM_DIR):
	mkdir -p $(SIM_DIR)

vectors:
	python3 scripts/generate_test_vectors.py

build: $(SIM_DIR) vectors
	$(IVERILOG) $(IVERILOG_FLAGS) -o $(SIM_BIN) $(TB_SRC) $(RTL_SRCS)

run_sim: build
	$(VVP) $(SIM_BIN)

wave: run_sim
	$(GTKWAVE) $(VCD_FILE) &

clean:
	rm -rf $(SIM_DIR) rtl/testbench/test_vectors.v

.PHONY: all build run_sim wave clean vectors
