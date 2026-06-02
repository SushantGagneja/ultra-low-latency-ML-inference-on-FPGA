`timescale 1ns/1ps

module microstructure_cosim_tb;

    reg clk;
    reg rst_n;
    
    // SPI signals
    reg sclk;
    reg cs_n;
    reg mosi;
    wire miso;
    
    // SPI Slave -> Tick Parser
    wire tick_start;
    wire [127:0] tick_payload;
    
    // Tick Parser -> OFI Engine
    wire tick_valid;
    wire [31:0] bid_price_q;
    wire [31:0] ask_price_q;
    wire [31:0] bid_qty_q;
    wire [31:0] ask_qty_q;
    
    // OFI Engine output
    wire ofi_valid;
    wire signed [31:0] ofi_q;
    
    // VWAP Engine output
    wire vwap_valid;
    wire [33:0] vwap_q;
    wire vwap_busy;
    
    // Latch VWAP for testbench reading
    reg vwap_valid_latched = 0;
    reg [33:0] vwap_q_latched = 0;
    
    always @(posedge clk) begin
        if (vwap_valid) begin
            vwap_valid_latched <= 1'b1;
            vwap_q_latched <= vwap_q;
        end
        if (tick_valid) begin
            vwap_valid_latched <= 1'b0; // clear on next tick
        end
    end
    
    // Instantiate SPI Slave
    spi_slave u_spi (
        .rst_n(rst_n),
        .sclk(sclk),
        .cs_n(cs_n),
        .mosi(mosi),
        .miso(miso),
        .sys_clk(clk),
        
        // Legacy outputs left floating
        .bnn_start(),
        .bnn_spike_vector(),
        .bram_we(),
        .bram_waddr(),
        .bram_wdata(),
        .bnn_decision(2'd0),
        
        // Phase 2 outputs
        .tick_start(tick_start),
        .tick_payload(tick_payload)
    );
    
    // Instantiate Tick Parser
    tick_parser u_parser (
        .clk(clk),
        .rst_n(rst_n),
        .tick_start(tick_start),
        .tick_payload(tick_payload),
        .tick_valid(tick_valid),
        .bid_price_q17_15(bid_price_q),
        .ask_price_q17_15(ask_price_q),
        .bid_qty_q16_16(bid_qty_q),
        .ask_qty_q16_16(ask_qty_q)
    );
    
    // Instantiate OFI Engine
    ofi_engine u_ofi (
        .clk(clk),
        .rst_n(rst_n),
        .tick_valid(tick_valid),
        .bid_price_q17_15(bid_price_q),
        .ask_price_q17_15(ask_price_q),
        .bid_qty_q16_16(bid_qty_q),
        .ask_qty_q16_16(ask_qty_q),
        .ofi_valid(ofi_valid),
        .ofi_q16_16(ofi_q)
    );
    
    // Instantiate VWAP Engine
    vwap_engine u_vwap (
        .clk(clk),
        .rst_n(rst_n),
        .tick_valid(tick_valid),
        .bid_price_q17_15(bid_price_q),
        .ask_price_q17_15(ask_price_q),
        .bid_qty_q16_16(bid_qty_q),
        .ask_qty_q16_16(ask_qty_q),
        .vwap_valid(vwap_valid),
        .vwap_q18_15(vwap_q),
        .vwap_busy(vwap_busy)
    );
    
    // Monitor write_ptr specifically for tick 21 wrap around
    always @(posedge clk) begin
        if (tick_valid) begin
            $display("TICK_RX: Write Ptr Before = %d", u_vwap.write_ptr);
        end
    end

    // Clock generation (100 MHz sys_clk)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // SPI Clock generation (40 MHz)
    // 25 ns period -> 12.5 ns half-period
    task spi_send_bit;
        input bit_val;
        begin
            mosi = bit_val;
            #12.5;
            sclk = 1;
            #12.5;
            sclk = 0;
        end
    endtask

    // VCD Dump
    initial begin
        $dumpfile("microstructure_cosim.vcd");
        $dumpvars(0, microstructure_cosim_tb);
    end

    // The Python runner will append test vectors below this module.
endmodule
