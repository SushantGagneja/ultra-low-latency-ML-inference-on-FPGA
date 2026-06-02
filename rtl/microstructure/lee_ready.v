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
    output reg  [1:0]   lr_class // 00 = Neutral, 01 = Buyer-Initiated, 11 = Seller-Initiated
);

    // TODO: Implement logic here

endmodule
