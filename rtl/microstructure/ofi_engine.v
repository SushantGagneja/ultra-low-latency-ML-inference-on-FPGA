`timescale 1ns/1ps

module ofi_engine (
    input  wire         clk,
    input  wire         rst_n,
    
    // Interface from Tick Parser
    input  wire         tick_valid,
    input  wire [31:0]  bid_price_q17_15,
    input  wire [31:0]  ask_price_q17_15,
    input  wire [31:0]  bid_qty_q16_16,
    input  wire [31:0]  ask_qty_q16_16,
    
    // Outputs to Feature Quantizer
    output reg          ofi_valid,
    output reg  signed [31:0] ofi_q16_16
);

    // State registers for previous tick
    reg [31:0] prev_bid_price;
    reg [31:0] prev_ask_price;
    reg [31:0] prev_bid_qty;
    reg [31:0] prev_ask_qty;
    reg        is_first_tick;

    reg signed [31:0] term1_v, term2_v, term3_v, term4_v;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ofi_valid      <= 1'b0;
            ofi_q16_16     <= 32'd0;
            prev_bid_price <= 32'd0;
            prev_ask_price <= 32'd0;
            prev_bid_qty   <= 32'd0;
            prev_ask_qty   <= 32'd0;
            is_first_tick  <= 1'b1;
        end else begin
            ofi_valid <= tick_valid; // Output valid 1 cycle after tick_valid
            
            if (tick_valid) begin
                if (is_first_tick) begin
                    ofi_q16_16 <= 32'd0;
                    is_first_tick <= 1'b0;
                end else begin
                    // Combinational CKS OFI logic
                    // bid_side = bid_size * I(bid >= prev) - prev_bid_size * I(bid <= prev)
                    // ask_side = ask_size * I(ask <= prev) - prev_ask_size * I(ask >= prev)
                    // OFI = bid_side - ask_side
                    
                    // Note: In hardware, multiplication by an indicator (1 or 0) 
                    // is just a multiplexer.
                    
                    term1_v = (bid_price_q17_15 >= prev_bid_price) ? bid_qty_q16_16 : 32'd0;
                    term2_v = (bid_price_q17_15 <= prev_bid_price) ? prev_bid_qty : 32'd0;
                    term3_v = (ask_price_q17_15 <= prev_ask_price) ? ask_qty_q16_16 : 32'd0;
                    term4_v = (ask_price_q17_15 >= prev_ask_price) ? prev_ask_qty : 32'd0;
                    
                    // Strict 32-bit 2's complement arithmetic
                    ofi_q16_16 <= term1_v - term2_v - term3_v + term4_v;
                end
                
                // Update state
                prev_bid_price <= bid_price_q17_15;
                prev_ask_price <= ask_price_q17_15;
                prev_bid_qty   <= bid_qty_q16_16;
                prev_ask_qty   <= ask_qty_q16_16;
            end
        end
    end

endmodule
