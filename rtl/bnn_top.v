`timescale 1ns/1ps

module bnn_top (
    input  wire sys_clk,
    input  wire rst_n,
    
    // SPI Slave Pins
    input  wire spi_sclk,
    input  wire spi_cs_n,
    input  wire spi_mosi,
    output wire spi_miso,
    
    // Interrupt / Done pin
    output wire fpga_done
);

    // Interconnects
    wire        bnn_start_legacy;
    wire [15:0] bnn_spike_legacy;
    
    // Legacy SPI paths
    wire        bram_we;
    wire [7:0]  bram_waddr;
    wire [7:0]  bram_wdata;
    
    wire [1:0]  bnn_decision;
    wire        bnn_done;

    // Microstructure Pipeline Wires
    wire        tick_start;
    wire [127:0] tick_payload;

    // SPI Slave
    spi_slave u_spi_slave (
        .rst_n(rst_n),
        .sclk(spi_sclk),
        .cs_n(spi_cs_n),
        .mosi(spi_mosi),
        .miso(spi_miso),
        
        .sys_clk(sys_clk),
        .bnn_start(bnn_start_legacy),
        .bnn_spike_vector(bnn_spike_legacy),
        
        .bram_we(bram_we),
        .bram_waddr(bram_waddr),
        .bram_wdata(bram_wdata),
        
        .bnn_decision(bnn_decision),
        
        // Microstructure outputs
        .tick_start(tick_start),
        .tick_payload(tick_payload)
    );
    
    wire tick_valid;
    wire [31:0] bid_price_q;
    wire [31:0] ask_price_q;
    wire [31:0] bid_qty_q;
    wire [31:0] ask_qty_q;
    
    wire ofi_valid;
    wire [31:0] ofi_q;
    
    wire vwap_valid;
    wire [33:0] vwap_q;
    
    wire lr_valid;
    wire [1:0] lr_class;
    
    wire spike_valid;
    wire [31:0] spike_vector;

    // 1. Tick Parser
    tick_parser u_parser (
        .clk(sys_clk),
        .rst_n(rst_n),
        .tick_start(tick_start),
        .tick_payload(tick_payload),
        .tick_valid(tick_valid),
        .bid_price_q17_15(bid_price_q),
        .ask_price_q17_15(ask_price_q),
        .bid_qty_q16_16(bid_qty_q),
        .ask_qty_q16_16(ask_qty_q)
    );

    // 2. Lee-Ready
    lee_ready u_lr (
        .clk(sys_clk),
        .rst_n(rst_n),
        .tick_valid(tick_valid),
        .bid_price_q17_15(bid_price_q),
        .ask_price_q17_15(ask_price_q),
        .lr_valid(lr_valid),
        .lr_class(lr_class)
    );
    
    // 3. OFI Engine
    ofi_engine u_ofi (
        .clk(sys_clk),
        .rst_n(rst_n),
        .tick_valid(tick_valid),
        .bid_price_q17_15(bid_price_q),
        .ask_price_q17_15(ask_price_q),
        .bid_qty_q16_16(bid_qty_q),
        .ask_qty_q16_16(ask_qty_q),
        .ofi_valid(ofi_valid),
        .ofi_q16_16(ofi_q)
    );
    
    // 4. VWAP Engine
    vwap_engine u_vwap (
        .clk(sys_clk),
        .rst_n(rst_n),
        .tick_valid(tick_valid),
        .bid_price_q17_15(bid_price_q),
        .ask_price_q17_15(ask_price_q),
        .bid_qty_q16_16(bid_qty_q),
        .ask_qty_q16_16(ask_qty_q),
        .vwap_valid(vwap_valid),
        .vwap_q18_15(vwap_q),
        .vwap_busy() // left unconnected intentionally
    );

    wire [31:0] midpoint = (bid_price_q + ask_price_q) >> 1;

    // 5. Hardware Quantizer
    hw_quantizer u_quantizer (
        .clk(sys_clk),
        .rst_n(rst_n),
        .tick_valid(tick_valid),
        .ofi_q16_16(ofi_q),
        .vwap_q18_15(vwap_q),
        .midpoint(midpoint),
        .lr_class(lr_class),
        .ofi_valid(ofi_valid),
        .vwap_valid(vwap_valid),
        .lr_valid(lr_valid),
        .spike_vector(spike_vector),
        .spike_valid(spike_valid)
    );

    // Arbitration Logic
    wire bnn_start = bnn_start_legacy | spike_valid;
    wire [31:0] bnn_spike_vector = bnn_start_legacy ? {16'd0, bnn_spike_legacy} : spike_vector;

`ifdef FORMAL
    // Formal verification: Arbitration Mutual Exclusion
    // Ensures that the legacy 0x01 SPI path and the 0x10 microstructure path
    // never attempt to trigger the BNN inference simultaneously.
    initial assume(!rst_n);
    always @(posedge sys_clk) begin
        if (rst_n && $past(rst_n)) begin
            assert (!(bnn_start_legacy & spike_valid));
        end
    end
`endif

    // Unrolled BNN Core
    bnn_core_unrolled u_bnn_core (
        .clk(sys_clk),
        .rst_n(rst_n),
        .start(bnn_start),
        .spike_vector(bnn_spike_vector), // Pass all 32 temporal bits
        .done(bnn_done),
        .decision(bnn_decision)
    );

    // Route done signal to external pin
    // Using a simple RS latch pattern or directly assigning since done is pulsed.
    // Wait, bnn_done is a 1-cycle pulse. The ESP32 is waiting for a level!
    // If we just output a pulse, the ESP32 GPIO polling might miss it.
    // We need to latch it.
    reg done_latch;
    reg [2:0] spi_cs_sync;
    wire spi_cs_falling;

    always @(posedge sys_clk or negedge rst_n) begin
        if (!rst_n) begin
            spi_cs_sync <= 3'b111;
        end else begin
            spi_cs_sync <= {spi_cs_sync[1:0], spi_cs_n};
        end
    end

    assign spi_cs_falling = (spi_cs_sync[2:1] == 2'b10);

    always @(posedge sys_clk or negedge rst_n) begin
        if (!rst_n) begin
            done_latch <= 1'b0;
        end else begin
            if (spi_cs_falling || bnn_start) begin
                done_latch <= 1'b0; // Clear when ESP32 starts any SPI transaction
            end else if (bnn_done) begin
                done_latch <= 1'b1; // Set when inference completes
            end
        end
    end
    
    assign fpga_done = done_latch;

endmodule
