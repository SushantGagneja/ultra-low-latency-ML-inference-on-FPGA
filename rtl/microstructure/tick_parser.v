`timescale 1ns/1ps

module tick_parser (
    input  wire         clk,
    input  wire         rst_n,
    
    input  wire         tick_start,
    input  wire         vwap_busy,
    input  wire [127:0] tick_payload,
    input  wire [7:0]   tick_metadata,
    
    // Outputs to Feature Engines
    output reg          tick_valid,
    output reg  [31:0]  bid_price_q17_15,
    output reg  [31:0]  ask_price_q17_15,
    output reg  [31:0]  bid_qty_q16_16,
    output reg  [31:0]  ask_qty_q16_16,
    output reg  [1:0]   velocity,
    output reg          regime_select
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tick_valid       <= 1'b0;
            bid_price_q17_15 <= 32'd0;
            ask_price_q17_15 <= 32'd0;
            bid_qty_q16_16   <= 32'd0;
            ask_qty_q16_16   <= 32'd0;
            velocity         <= 2'd0;
            regime_select    <= 1'b0;
        end else begin
            // 1-cycle strobe, only valid if VWAP engine is ready
            tick_valid <= tick_start & ~vwap_busy;
            
            if (tick_start && !vwap_busy) begin
                // Update latched state only if pipeline is ready 128-bit payload
                // [127:96]  Bid Price
                // [95:64]   Bid Qty
                // [63:32]   Ask Price
                // [31:0]    Ask Qty
                bid_price_q17_15 <= tick_payload[127:96];
                bid_qty_q16_16   <= tick_payload[95:64];
                ask_price_q17_15 <= tick_payload[63:32];
                ask_qty_q16_16   <= tick_payload[31:0];
                velocity         <= tick_metadata[1:0];
                regime_select    <= tick_metadata[2];
            end
        end
    end

endmodule
