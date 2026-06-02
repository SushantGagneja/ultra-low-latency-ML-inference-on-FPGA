`timescale 1ns/1ps

module hw_quantizer (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         tick_valid,

    // Feature inputs
    input  wire signed [31:0] ofi_q16_16,   // signed Q16.16
    input  wire        [33:0] vwap_q18_15,  // unsigned Q18.15
    input  wire        [31:0] midpoint,      // Q17.15, from tick_parser
    input  wire        [1:0]  lr_class,      // LR_BUYER/SELLER/NEUTRAL/UNDEF
    input  wire               ofi_valid,
    input  wire               vwap_valid,
    input  wire               lr_valid,

    // Output
    output reg  [7:0] spike_vector,
    output reg        spike_valid
);

    reg ofi_valid_reg;
    reg vwap_valid_reg;
    reg lr_valid_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ofi_valid_reg  <= 1'b0;
            vwap_valid_reg <= 1'b0;
            lr_valid_reg   <= 1'b0;
        end else begin
            if (tick_valid) begin
                ofi_valid_reg  <= 1'b0;
                vwap_valid_reg <= 1'b0;
                lr_valid_reg   <= 1'b0;
            end else begin
                if (ofi_valid)  ofi_valid_reg  <= 1'b1;
                if (vwap_valid) vwap_valid_reg <= 1'b1;
                if (lr_valid)   lr_valid_reg   <= 1'b1;
            end
        end
    end

    wire all_features_ready = (ofi_valid | ofi_valid_reg) & 
                              (vwap_valid | vwap_valid_reg) & 
                              (lr_valid | lr_valid_reg);

    reg all_features_ready_prev;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) all_features_ready_prev <= 1'b0;
        else all_features_ready_prev <= all_features_ready;
    end
    
    wire trigger_quantization = all_features_ready & ~all_features_ready_prev;

    wire [32:0] midpoint_ext = {1'b0, midpoint};
    wire [32:0] vwap_aligned = vwap_q18_15[33:1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spike_vector <= 16'd0;
            spike_valid  <= 1'b0;
        end else begin
            spike_valid <= 1'b0;
            if (trigger_quantization) begin
                spike_vector[0] <= (ofi_q16_16 > 32'sd327680);
                spike_vector[1] <= (ofi_q16_16 > 32'sd0);
                spike_vector[2] <= (ofi_q16_16 < -32'sd327680);
                spike_vector[3] <= (ofi_q16_16 < 32'sd0);
                spike_vector[4] <= (midpoint_ext > vwap_aligned);
                spike_vector[5] <= (midpoint_ext < vwap_aligned);
                spike_vector[6] <= (lr_class == 2'b01);
                spike_vector[7] <= (lr_class == 2'b10);
                spike_valid <= 1'b1;
            end
        end
    end

endmodule
