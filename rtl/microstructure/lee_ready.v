`timescale 1ns/1ps

/*
 * Lee-Ready Proxy Classifier (OFI / Microstructure Engine)
 *
 * This module implements a quote-driven proxy of the Lee and Ready (1991) 
 * trade classification algorithm.
 *
 * ACADEMIC CITATION & PROXY JUSTIFICATION:
 * True Lee-Ready classification (Lee & Ready, "Inferring Trade Direction from 
 * Intraday Data", Journal of Finance, 1991) requires explicit transaction 
 * prices (e.g. from an aggTrade stream) to compare against the prevailing 
 * National Best Bid and Offer (NBBO) midpoint. 
 * 
 * Since this microstructure engine operates strictly on Level 1 LOB data 
 * (Binance bookTicker: best bid/ask prices and quantities without trades), 
 * we implement the academically documented quote-based proxy:
 * 
 * Classification Rules:
 *   - Buyer-Initiated (Class 1): New Midpoint > Previous Midpoint
 *   - Seller-Initiated (Class -1): New Midpoint < Previous Midpoint
 *   - Neutral / Unchanged (Class 0): New Midpoint == Previous Midpoint
 *
 * This proxy assumes that aggressive market orders that consume liquidity 
 * at the best bid/ask will systematically shift the midpoint in the direction 
 * of the aggression.
 */

module lee_ready (
    input  wire         clk,
    input  wire         rst_n,
    
    // Interface from Tick Parser
    input  wire         tick_valid,
    input  wire [31:0]  bid_price_q17_15,
    input  wire [31:0]  ask_price_q17_15,
    
    // Outputs to Feature Aggregator
    output reg          lr_valid,
    output reg  [1:0]   lr_class // 00 = Neutral, 01 = Buyer-Initiated, 10 = Seller-Initiated, 11 = Undefined
);

    // Encoding matching Python golden model
    localparam LR_NEUTRAL = 2'b00;
    localparam LR_BUYER   = 2'b01;
    localparam LR_SELLER  = 2'b10;
    localparam LR_UNDEF   = 2'b11;

    // Internal State
    reg [31:0] midpoint_prev;
    reg        tick_seen;
    
    // Combinational logic for current midpoint
    // TRUNCATION NOTE: 
    // The midpoint computation (bid_price + ask_price) >> 1 is done in Q17.15 
    // arithmetic with the shift truncating the LSB, not rounding. If the sum 
    // is odd, the 0.5 is discarded. This perfectly matches the Python golden 
    // model `to_unsigned((bp + ap) >> 1, 32)` which also truncates.
    wire [31:0] midpoint_curr = (bid_price_q17_15 + ask_price_q17_15) >> 1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lr_valid      <= 1'b0;
            lr_class      <= LR_UNDEF;
            midpoint_prev <= 32'd0;
            tick_seen     <= 1'b0;
        end else if (tick_valid) begin
            // 1. Register current midpoint as prev for the NEXT tick
            midpoint_prev <= midpoint_curr;
            
            // 2. Mark that we've seen at least one tick
            tick_seen <= 1'b1;
            
            // 3. Compute classification and output
            if (!tick_seen) begin
                // First tick: output is UNDEF, valid is False
                lr_valid <= 1'b0;
                lr_class <= LR_UNDEF;
            end else begin
                // Second tick onward: Valid comparison against midpoint_prev
                lr_valid <= 1'b1;
                
                if (midpoint_curr > midpoint_prev) begin
                    lr_class <= LR_BUYER;
                end else if (midpoint_curr < midpoint_prev) begin
                    lr_class <= LR_SELLER;
                end else begin
                    lr_class <= LR_NEUTRAL;
                end
            end
        end else begin
            // Clear valid strobe on cycles without a tick
            // (Assuming downstream expects a 1-cycle valid strobe matching OFI)
            lr_valid <= 1'b0;
        end
    end

endmodule
