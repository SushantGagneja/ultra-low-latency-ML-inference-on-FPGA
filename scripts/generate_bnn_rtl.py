import numpy as np
from pathlib import Path
import math

def generate_bnn_rtl(w1, w2, b1_A, b2_A, b1_B, b2_B, output_file="rtl/bnn_core_unrolled.v"):
    """
    w1: shape (32, 40) - 32 hidden neurons, 40 inputs. Values in {-1, 0, 1}
    w2: shape (3, 32) - 3 output neurons, 32 inputs. Values in {-1, 0, 1}
    b1_A/B: shape (32,) - bias for hidden layer
    b2_A/B: shape (3,) - bias for output layer
    """
    num_inputs = w1.shape[1]
    num_hidden = w1.shape[0]
    num_outputs = w2.shape[0]
    
    with open(output_file, 'w') as f:
        f.write("`timescale 1ns/1ps\n\n")
        f.write("module bnn_core_unrolled (\n")
        f.write("    input  wire        clk,\n")
        f.write("    input  wire        rst_n,\n")
        f.write("    input  wire        start,\n")
        f.write("    input  wire [39:0] spike_vector,\n")
        f.write("    input  wire        regime_select,\n")
        f.write("    output reg         done,\n")
        f.write("    output reg  [1:0]  decision\n")
        f.write(");\n\n")
        
        # We need a 1-stage pipeline (2 cycles total)
        # Clock 1: Calculate hidden layer and store in register.
        # Clock 2: Calculate output layer and raise done.
        
        f.write("    // --- LAYER 1: 40 Inputs -> 32 Hidden (Combinatorial) ---\n")
        
        for j in range(num_hidden):
            terms = []
            num_nonzero = 0
            for i in range(num_inputs):
                if w1[j, i] == 1:
                    terms.append(f"spike_vector[{i}]")
                    num_nonzero += 1
                elif w1[j, i] == -1:
                    terms.append(f"(~spike_vector[{i}])")
                    num_nonzero += 1
                    
            if num_nonzero == 0:
                f.write(f"    wire hidden_comb_{j} = 1'b0; // Pruned\n")
            else:
                threshold_A = math.ceil((num_nonzero - b1_A[j]) / 2.0)
                threshold_B = math.ceil((num_nonzero - b1_B[j]) / 2.0)
                bit_width = max(1, math.ceil(math.log2(num_nonzero + 1)))
                
                sum_expr = " + ".join(terms)
                f.write(f"    wire [{bit_width-1}:0] sum_h_{j} = {sum_expr};\n")
                f.write(f"    wire hidden_comb_{j} = (sum_h_{j} >= (regime_select ? $signed({threshold_A}) : $signed({threshold_B}))) ? 1'b1 : 1'b0;\n")
                
        f.write("\n    // --- PIPELINE REGISTER ---\n")
        f.write(f"    reg [{num_hidden-1}:0] hidden_reg;\n")
        f.write("    reg stage1_valid;\n")
        f.write("    always @(posedge clk or negedge rst_n) begin\n")
        f.write("        if (!rst_n) begin\n")
        f.write(f"            hidden_reg <= {num_hidden}'d0;\n")
        f.write("            stage1_valid <= 1'b0;\n")
        f.write("        end else begin\n")
        f.write("            stage1_valid <= start;\n")
        f.write("            if (start) begin\n")
        f.write("                hidden_reg <= { ")
        # Pack bits from 31 down to 0
        h_bits = [f"hidden_comb_{j}" for j in range(num_hidden-1, -1, -1)]
        f.write(", ".join(h_bits))
        f.write(" };\n")
        f.write("            end\n")
        f.write("        end\n")
        f.write("    end\n\n")
        
        f.write("    // --- LAYER 2: 32 Hidden -> 1 Output (Combinatorial) ---\n")
        for j in range(num_outputs):
            terms = []
            num_nonzero = 0
            for i in range(num_hidden):
                if w2[j, i] == 1:
                    terms.append(f"hidden_reg[{i}]")
                    num_nonzero += 1
                elif w2[j, i] == -1:
                    terms.append(f"(~hidden_reg[{i}])")
                    num_nonzero += 1
                    
            if num_nonzero == 0:
                f.write(f"    wire out_comb_{j} = 1'b0; // Pruned\n")
            else:
                threshold_A = math.ceil((num_nonzero - b2_A[j]) / 2.0)
                threshold_B = math.ceil((num_nonzero - b2_B[j]) / 2.0)
                bit_width = max(1, math.ceil(math.log2(num_nonzero + 1)))
                
                sum_expr = " + ".join(terms)
                f.write(f"    wire [{bit_width-1}:0] sum_o_{j} = {sum_expr};\n")
                f.write(f"    wire out_comb_{j} = (sum_o_{j} >= (regime_select ? $signed({threshold_A}) : $signed({threshold_B}))) ? 1'b1 : 1'b0;\n")
                
        f.write("\n    // --- FINAL OUTPUT REGISTER ---\n")
        f.write("    always @(posedge clk or negedge rst_n) begin\n")
        f.write("        if (!rst_n) begin\n")
        f.write("            done <= 1'b0;\n")
        f.write("            decision <= 2'b01; // Default HOLD\n")
        f.write("        end else begin\n")
        f.write("            done <= stage1_valid;\n")
        f.write("            if (stage1_valid) begin\n")
        f.write("                if (out_comb_0) begin\n")
        f.write("                    decision <= 2'b00; // BUY\n")
        f.write("                end else if (out_comb_2) begin\n")
        f.write("                    decision <= 2'b10; // SELL\n")
        f.write("                end else begin\n")
        f.write("                    decision <= 2'b01; // HOLD\n")
        f.write("                end\n")
        f.write("            end else begin\n")
        f.write("                done <= 1'b0;\n")
        f.write("            end\n")
        f.write("        end\n")
        f.write("    end\n\n")
        
        f.write("endmodule\n")

if __name__ == "__main__":
    weights_path = Path("fpga_weights/ternary_weights.npz")
    if not weights_path.exists():
        print(f"Error: {weights_path} not found.")
        exit(1)
        
    data = np.load(weights_path)
    w1 = data['w1']
    w2 = data['w2']
    b1_A = data['b1_A']
    b2_A = data['b2_A']
    b1_B = data['b1_B']
    b2_B = data['b2_B']
    
    print(f"Loaded Layer 1 Weights: {w1.shape}, non-zero: {np.count_nonzero(w1)}")
    print(f"Loaded Layer 2 Weights: {w2.shape}, non-zero: {np.count_nonzero(w2)}")
    
    output_file = Path("rtl/bnn_core_unrolled.v")
    generate_bnn_rtl(w1, w2, b1_A, b2_A, b1_B, b2_B, str(output_file))
    print(f"Generated fully unrolled sparse BNN RTL at {output_file}")
