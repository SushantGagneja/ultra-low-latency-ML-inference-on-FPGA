`timescale 1ns/1ps

/*
 * Synchronous BRAM for Microstructure Pipeline (VWAP Ring Buffer)
 *
 * This module infers a synchronous read, synchronous write RAM.
 * Synthesis directive `(* ram_style = "block" *)` enforces block RAM inference.
 * Read-before-write or Write-first behavior is often synthesis-dependent.
 * The Phase 2.2 pipeline explicitly separates BRAM read and BRAM write
 * into distinct cycles to guarantee correctness regardless of read-first
 * or write-first synthesis behavior.
 *
 * Port width: 120 bits (72-bit sum_pv + 48-bit sum_v)
 * Depth: 32 entries (20 ticks rounded up to nearest power of 2)
 */

module bram_microstructure (
    input  wire         clk,
    input  wire         we,
    input  wire [4:0]   addr,
    input  wire [119:0] din,
    output reg  [119:0] dout
);

    // Enforce Block RAM inference
    (* ram_style = "block" *) reg [119:0] ram [0:31];

    // Initialize to zero for simulation and FPGA bitstream
    integer i;
    initial begin
        for (i = 0; i < 32; i = i + 1) begin
            ram[i] = 120'd0;
        end
    end

    always @(posedge clk) begin
        if (we) begin
            ram[addr] <= din;
        end
        // Synchronous read (1-cycle latency)
        dout <= ram[addr];
    end

endmodule
