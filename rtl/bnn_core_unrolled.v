`timescale 1ns/1ps

module bnn_core_unrolled (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [7:0]  spike_vector,
    output reg         done,
    output reg  [1:0]  decision
);

    // --- LAYER 1: 8 Inputs -> 32 Hidden (Combinatorial) ---
    wire hidden_comb_0 = 1'b0; // Pruned
    wire hidden_comb_1 = 1'b0; // Pruned
    wire hidden_comb_2 = 1'b0; // Pruned
    wire hidden_comb_3 = 1'b0; // Pruned
    wire hidden_comb_4 = 1'b0; // Pruned
    wire hidden_comb_5 = 1'b0; // Pruned
    wire hidden_comb_6 = 1'b0; // Pruned
    wire hidden_comb_7 = 1'b0; // Pruned
    wire hidden_comb_8 = 1'b0; // Pruned
    wire hidden_comb_9 = 1'b0; // Pruned
    wire hidden_comb_10 = 1'b0; // Pruned
    wire hidden_comb_11 = 1'b0; // Pruned
    wire hidden_comb_12 = 1'b0; // Pruned
    wire hidden_comb_13 = 1'b0; // Pruned
    wire hidden_comb_14 = 1'b0; // Pruned
    wire hidden_comb_15 = 1'b0; // Pruned
    wire hidden_comb_16 = 1'b0; // Pruned
    wire [2:0] sum_h_17 = spike_vector[0] + spike_vector[1] + (~spike_vector[3]) + spike_vector[4] + (~spike_vector[5]) + spike_vector[6] + (~spike_vector[7]);
    wire hidden_comb_17 = (sum_h_17 >= 4) ? 1'b1 : 1'b0;
    wire hidden_comb_18 = 1'b0; // Pruned
    wire hidden_comb_19 = 1'b0; // Pruned
    wire hidden_comb_20 = 1'b0; // Pruned
    wire hidden_comb_21 = 1'b0; // Pruned
    wire hidden_comb_22 = 1'b0; // Pruned
    wire hidden_comb_23 = 1'b0; // Pruned
    wire hidden_comb_24 = 1'b0; // Pruned
    wire hidden_comb_25 = 1'b0; // Pruned
    wire hidden_comb_26 = 1'b0; // Pruned
    wire hidden_comb_27 = 1'b0; // Pruned
    wire hidden_comb_28 = 1'b0; // Pruned
    wire [2:0] sum_h_29 = spike_vector[1] + (~spike_vector[3]) + spike_vector[4] + (~spike_vector[5]) + spike_vector[6] + (~spike_vector[7]);
    wire hidden_comb_29 = (sum_h_29 >= 3) ? 1'b1 : 1'b0;
    wire hidden_comb_30 = 1'b0; // Pruned
    wire hidden_comb_31 = 1'b0; // Pruned

    // --- PIPELINE REGISTER ---
    reg [31:0] hidden_reg;
    reg stage1_valid;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hidden_reg <= 32'd0;
            stage1_valid <= 1'b0;
        end else begin
            stage1_valid <= start;
            if (start) begin
                hidden_reg <= { hidden_comb_31, hidden_comb_30, hidden_comb_29, hidden_comb_28, hidden_comb_27, hidden_comb_26, hidden_comb_25, hidden_comb_24, hidden_comb_23, hidden_comb_22, hidden_comb_21, hidden_comb_20, hidden_comb_19, hidden_comb_18, hidden_comb_17, hidden_comb_16, hidden_comb_15, hidden_comb_14, hidden_comb_13, hidden_comb_12, hidden_comb_11, hidden_comb_10, hidden_comb_9, hidden_comb_8, hidden_comb_7, hidden_comb_6, hidden_comb_5, hidden_comb_4, hidden_comb_3, hidden_comb_2, hidden_comb_1, hidden_comb_0 };
            end
        end
    end

    // --- LAYER 2: 32 Hidden -> 1 Output (Combinatorial) ---
    wire [1:0] sum_o_0 = hidden_reg[17] + hidden_reg[29];
    wire out_comb_0 = (sum_o_0 >= 1) ? 1'b1 : 1'b0;

    // --- FINAL OUTPUT REGISTER ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            decision <= 2'b01; // Default HOLD
        end else begin
            done <= stage1_valid;
            if (stage1_valid) begin
                // out_comb_0 == 1 -> BUY, 0 -> SELL
                decision <= out_comb_0 ? 2'b00 : 2'b10;
            end else begin
                done <= 1'b0;
            end
        end
    end

endmodule
