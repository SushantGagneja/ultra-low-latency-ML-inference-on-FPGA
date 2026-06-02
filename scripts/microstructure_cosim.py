#!/usr/bin/env python3
import csv
import subprocess
import sys
from pathlib import Path
from golden_ofi import GoldenOFI, float_to_q17_15, float_to_q16_16, to_signed_32
from golden_vwap import GoldenVWAP

ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "sim"
TB_SRC = ROOT / "rtl" / "testbench" / "microstructure_cosim_tb.v"

def generate_test_vectors(csv_path: Path, limit: int = 1000):
    from golden_lee_ready import GoldenLeeReady, to_unsigned
    from golden_quantizer import GoldenQuantizer
    
    ticks = []
    golden_ofi = GoldenOFI()
    golden_vwap = GoldenVWAP()
    golden_lr = GoldenLeeReady()
    golden_q = GoldenQuantizer()
    golden_outputs = []
    golden_vwap_outputs = []
    golden_lr_outputs = []
    golden_q_outputs = []
    
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if len(ticks) >= limit:
                break
            bid = float(row[1])
            bid_qty = float(row[2])
            ask = float(row[3])
            ask_qty = float(row[4])
            
            bp = float_to_q17_15(bid)
            ap = float_to_q17_15(ask)
            bq = float_to_q16_16(bid_qty)
            aq = float_to_q16_16(ask_qty)
            
            ofi = golden_ofi.compute_ofi(bp, ap, bq, aq)
            v_valid, vwap = golden_vwap.update(bp, ap, bq, aq)
            lr_valid, lr_class = golden_lr.update(bp, ap)
            
            midpoint = to_unsigned((bp + ap) >> 1, 32)
            q_valid, spike_vector = golden_q.update(
                ofi_q16_16=ofi,
                vwap_q18_15=vwap,
                midpoint=midpoint,
                lr_class=lr_class,
                ofi_valid=True, # ofi logic doesn't explicitly return valid flag in current script
                vwap_valid=v_valid,
                lr_valid=lr_valid
            )
            
            ticks.append((bp, bq, ap, aq))
            golden_outputs.append(ofi)
            golden_vwap_outputs.append((v_valid, vwap))
            golden_lr_outputs.append((lr_valid, lr_class))
            golden_q_outputs.append((q_valid, spike_vector))
            
    print(f"Generated {len(ticks)} test vectors.")
    return ticks, golden_outputs, golden_vwap_outputs, golden_lr_outputs, golden_q_outputs

def write_testbench(ticks: list):
    vector_file = ROOT / "rtl" / "testbench" / "microstructure_vectors.v"
    
    with open(vector_file, "w") as f:
        f.write("`timescale 1ns/1ps\n")
        f.write("module microstructure_vectors;\n")
        f.write("    // This module is injected into the testbench\n")
        f.write("    initial begin\n")
        f.write("        microstructure_cosim_tb.rst_n = 0;\n")
        f.write("        microstructure_cosim_tb.sclk = 0;\n")
        f.write("        microstructure_cosim_tb.cs_n = 1;\n")
        f.write("        microstructure_cosim_tb.mosi = 0;\n")
        f.write("        #100 microstructure_cosim_tb.rst_n = 1;\n")
        f.write("        #100;\n")
        
        for i, (bp, bq, ap, aq) in enumerate(ticks):
            # 136 bit payload: 8-ctrl, 32-bp, 32-bq, 32-ap, 32-aq
            # Control is 0x10
            payload = (0x10 << 128) | (bp << 96) | (bq << 64) | (ap << 32) | aq
            
            f.write(f"        // Tick {i}\n")
            f.write("        microstructure_cosim_tb.cs_n = 0;\n")
            f.write("        #25;\n")
            for bit in range(135, -1, -1):
                bit_val = (payload >> bit) & 1
                f.write(f"        microstructure_cosim_tb.spi_send_bit({bit_val});\n")
            f.write("        #25;\n")
            f.write("        microstructure_cosim_tb.cs_n = 1;\n")
            f.write("        #500; // wait for valid pulse\n")
            f.write("        $display(\"OFI_OUT: %d\", microstructure_cosim_tb.ofi_q);\n")
            f.write("        $display(\"VWAP_VALID: %d\", microstructure_cosim_tb.vwap_valid_latched);\n")
            f.write("        if (microstructure_cosim_tb.vwap_valid_latched) begin\n")
            f.write("            $display(\"VWAP_OUT: %d\", microstructure_cosim_tb.vwap_q_latched);\n")
            f.write("        end\n")
            
            f.write("        $display(\"LR_VALID: %d\", microstructure_cosim_tb.lr_valid_latched);\n")
            f.write("        if (microstructure_cosim_tb.lr_valid_latched) begin\n")
            f.write("            $display(\"LR_OUT: %d\", microstructure_cosim_tb.lr_class_latched);\n")
            f.write("        end\n")
            
            f.write("        $display(\"SPIKE_VALID: %d\", microstructure_cosim_tb.spike_valid_latched);\n")
            f.write("        if (microstructure_cosim_tb.spike_valid_latched) begin\n")
            f.write("            $display(\"SPIKE_OUT: %d\", microstructure_cosim_tb.spike_vector_latched);\n")
            f.write("        end\n")
            
        f.write("        $finish;\n")
        f.write("    end\n")
        f.write("endmodule\n")
        
    return vector_file

def run_rtl_cosim(vector_file: Path):
    SIM_DIR.mkdir(exist_ok=True)
    sim_bin = SIM_DIR / "microstructure_cosim.vvp"
    
    cmd = [
        "iverilog", "-Wall", "-g2012",
        "-I", str(ROOT / "rtl"),
        "-I", str(ROOT / "rtl" / "microstructure"),
        "-o", str(sim_bin),
        str(TB_SRC),
        str(vector_file),
        str(ROOT / "rtl" / "spi_slave.v"),
        str(ROOT / "rtl" / "microstructure" / "tick_parser.v"),
        str(ROOT / "rtl" / "microstructure" / "ofi_engine.v"),
        str(ROOT / "rtl" / "microstructure" / "bram_microstructure.v"),
        str(ROOT / "rtl" / "microstructure" / "restoring_divider.v"),
        str(ROOT / "rtl" / "microstructure" / "vwap_engine.v"),
        str(ROOT / "rtl" / "microstructure" / "lee_ready.v"),
        str(ROOT / "rtl" / "microstructure" / "hw_quantizer.v"),
    ]
    
    subprocess.run(cmd, check=True)
    
    res = subprocess.run(["vvp", str(sim_bin)], capture_output=True, text=True, check=True)
    
    rtl_ofi = []
    rtl_vwap = []
    rtl_lr = []
    rtl_q = []
    for line in res.stdout.splitlines():
        if line.startswith("OFI_OUT:"):
            val = int(line.split(":")[1].strip())
            rtl_ofi.append(to_signed_32(val))
        elif line.startswith("VWAP_VALID:"):
            val = int(line.split(":")[1].strip())
            rtl_vwap.append({"valid": val == 1, "vwap": 0, "sum_pv": 0, "sum_v": 0})
        elif line.startswith("VWAP_OUT:"):
            val = int(line.split(":")[1].strip())
            rtl_vwap[-1]["vwap"] = val
        elif line.startswith("LR_VALID:"):
            val = int(line.split(":")[1].strip())
            rtl_lr.append({"valid": val == 1, "class": 3})
        elif line.startswith("LR_OUT:"):
            val = int(line.split(":")[1].strip())
            rtl_lr[-1]["class"] = val
        elif line.startswith("SPIKE_VALID:"):
            val = int(line.split(":")[1].strip())
            rtl_q.append({"valid": val == 1, "spike": 0})
        elif line.startswith("SPIKE_OUT:"):
            val = int(line.split(":")[1].strip())
            rtl_q[-1]["spike"] = val
            
    return rtl_ofi, rtl_vwap, rtl_lr, rtl_q

def main():
    csv_path = ROOT / "BTCUSDT-bookTicker-2024-01.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    ticks, golden_outputs, golden_vwap_outputs, golden_lr_outputs, golden_q_outputs = generate_test_vectors(csv_path, limit=1000)
    vector_file = write_testbench(ticks)
    rtl_ofi, rtl_vwap, rtl_lr, rtl_q = run_rtl_cosim(vector_file)
    
    if len(golden_outputs) != len(rtl_ofi):
        print(f"Mismatch in OFI count! Golden: {len(golden_outputs)}, RTL: {len(rtl_ofi)}")
        sys.exit(1)
        
    mismatches = 0
    for i in range(len(golden_outputs)):
        if golden_outputs[i] != rtl_ofi[i]:
            print(f"Tick {i} OFI Mismatch! Golden: {golden_outputs[i]}, RTL: {rtl_ofi[i]}")
            mismatches += 1
            
        g_valid, g_vwap = golden_vwap_outputs[i]
        r_vwap_dict = rtl_vwap[i]
        r_valid, r_vwap = r_vwap_dict["valid"], r_vwap_dict["vwap"]
        
        if g_valid != r_valid:
            print(f"Tick {i} VWAP VALID Mismatch! Golden: {g_valid}, RTL: {r_valid}")
            mismatches += 1
            
        if g_valid and r_valid and g_vwap != r_vwap:
            print(f"Tick {i} VWAP OUT Mismatch! Golden: {g_vwap}, RTL: {r_vwap}")
            mismatches += 1
            
        g_lr_valid, g_lr_class = golden_lr_outputs[i]
        r_lr_dict = rtl_lr[i]
        r_lr_valid, r_lr_class = r_lr_dict["valid"], r_lr_dict["class"]
        
        if g_lr_valid != r_lr_valid:
            print(f"Tick {i} LR VALID Mismatch! Golden: {g_lr_valid}, RTL: {r_lr_valid}")
            mismatches += 1
            
        if g_lr_valid and r_lr_valid and g_lr_class != r_lr_class:
            print(f"Tick {i} LR OUT Mismatch! Golden: {g_lr_class}, RTL: {r_lr_class}")
            mismatches += 1
            
        g_q_valid, g_spike = golden_q_outputs[i]
        r_q_dict = rtl_q[i]
        r_q_valid, r_spike = r_q_dict["valid"], r_q_dict["spike"]
        
        # Quantizer Step 3: verify spike_valid is NEVER asserted if not valid
        # Warmup domination
        if i < 19:
            if r_q_valid:
                print(f"Tick {i} SPIKE VALID asserted prematurely! Should be warmup.")
                mismatches += 1
                
        if g_q_valid != r_q_valid:
            print(f"Tick {i} SPIKE VALID Mismatch! Golden: {g_q_valid}, RTL: {r_q_valid}")
            mismatches += 1
            
        if g_q_valid and r_q_valid and g_spike != r_spike:
            print(f"Tick {i} SPIKE OUT Mismatch! Golden: {bin(g_spike)}, RTL: {bin(r_spike)}")
            mismatches += 1
            
        if mismatches >= 10:
            print("Too many mismatches, aborting...")
            sys.exit(1)
    

if __name__ == "__main__":
    main()
