import os
import numpy as np

def generate_vectors(num_vectors=100):
    weights_path = "../fpga_weights/weights.mem"
    if not os.path.exists(weights_path):
        weights_path = "fpga_weights/weights.mem"
        
    with open(weights_path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("//")]
        
    if len(lines) != 1216:
        raise ValueError(f"Expected 1216 weight bits, got {len(lines)}")
        
    weights = np.array([int(x) for x in lines])
    
    # Reconstruct w1 (16x64) and w2 (64x3)
    w1_bin = np.zeros((16, 64), dtype=int)
    for j in range(64):
        for i in range(16):
            w1_bin[i, j] = weights[j * 16 + i]
            
    w2_bin = np.zeros((64, 3), dtype=int)
    for j in range(3):
        for i in range(64):
            w2_bin[i, j] = weights[1024 + j * 64 + i]
            
    with open("rtl/testbench/test_vectors.v", "w") as tb_out:
        tb_out.write("// Auto-generated test vectors\n")
        tb_out.write(f"localparam NUM_VECTORS = {num_vectors};\n")
        tb_out.write("reg [15:0] test_inputs [0:NUM_VECTORS-1];\n")
        tb_out.write("reg [1:0]  test_outputs [0:NUM_VECTORS-1];\n\n")
        tb_out.write("initial begin\n")
        
        np.random.seed(42)
        for v_idx in range(num_vectors):
            # Generate random 16-bit vector
            v_bin = np.random.randint(0, 2, size=16)
            
            # Layer 1
            l1_out = np.zeros(64, dtype=int)
            for j in range(64):
                w = w1_bin[:, j]
                xnor = 1 - np.bitwise_xor(v_bin, w)
                pop = np.sum(xnor)
                l1_out[j] = 1 if pop >= 8 else 0
                
            # Layer 2
            l2_out = np.zeros(3, dtype=int)
            for j in range(3):
                w = w2_bin[:, j]
                xnor = 1 - np.bitwise_xor(l1_out, w)
                l2_out[j] = np.sum(xnor)
                
            # Argmax (winner take all)
            # 0 = BUY, 1 = HOLD, 2 = SELL
            prediction = int(np.argmax(l2_out))
            
            # Format for Verilog
            hex_input = "".join([str(x) for x in reversed(v_bin)])
            hex_val = int(hex_input, 2)
            
            tb_out.write(f"    test_inputs[{v_idx}] = 16'h{hex_val:04X};\n")
            tb_out.write(f"    test_outputs[{v_idx}] = 2'd{prediction};\n")
            
        tb_out.write("end\n")
    print(f"Generated {num_vectors} test vectors in rtl/testbench/test_vectors.v")

if __name__ == "__main__":
    generate_vectors()
