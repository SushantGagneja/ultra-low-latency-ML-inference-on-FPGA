#!/usr/bin/env python3
import sys
from collections import deque

def to_unsigned(val: int, bits: int) -> int:
    mask = (1 << bits) - 1
    return val & mask

class GoldenVWAP:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.tick_count = 0
        
        # Ring buffer for p*v (72 bits) and v (48 bits)
        self.pv_buffer = deque(maxlen=window_size)
        self.v_buffer = deque(maxlen=window_size)
        
        # Accumulators
        self.sum_pv = 0  # 72 bits
        self.sum_v = 0   # 48 bits

    def update(self, bid_price_q: int, ask_price_q: int, bid_qty_q: int, ask_qty_q: int):
        """
        Input: 
        - bid_price_q, ask_price_q (Q17.15): 32 bits
        - bid_qty_q, ask_qty_q (Q16.16): 32 bits
        
        Returns:
        - vwap_valid (bool)
        - vwap_q (int, Q18.15, 34 bits)
        """
        # Cycle 0: Calculate current tick values
        # price = (bid + ask) >> 1
        price_q = to_unsigned((bid_price_q + ask_price_q) >> 1, 32)
        volume_q = to_unsigned(bid_qty_q + ask_qty_q, 32)
        
        new_pv = to_unsigned(price_q * volume_q, 72)
        new_v = volume_q
        
        # Cycle 1: Read eviction values, Update Accumulators, Write new values
        if self.tick_count >= self.window_size:
            oldest_pv = self.pv_buffer[0]
            oldest_v = self.v_buffer[0]
        else:
            oldest_pv = 0
            oldest_v = 0
            
        self.sum_pv = to_unsigned(self.sum_pv - oldest_pv + new_pv, 72)
        self.sum_v = to_unsigned(self.sum_v - oldest_v + new_v, 48)
        
        self.pv_buffer.append(new_pv)
        self.v_buffer.append(new_v)
        
        if self.tick_count < self.window_size:
            self.tick_count += 1
            
        # Cycle 2: Output generation
        # Warmup guard
        if self.tick_count < self.window_size:
            return False, 0
            
        # Zero-division guard
        if self.sum_v == 0:
            return True, 0
            
        # Division: Q38.31 / Q20.16 = Q18.15
        vwap_q = self.sum_pv // self.sum_v
        vwap_q = to_unsigned(vwap_q, 34)
        
        # DEBUG
        print(f"GOLDEN TICK: sum_pv={self.sum_pv}, sum_v={self.sum_v}, vwap={vwap_q}")
        
        return True, vwap_q

def test_manual_ticks():
    print("Testing Golden VWAP with manual scenarios...")
    vwap_engine = GoldenVWAP(window_size=20)
    
    # Helper to convert float to fixed-point
    def float_to_q17_15(val: float) -> int:
        return int(val * (1 << 15)) & 0xFFFFFFFF
        
    def float_to_q16_16(val: float) -> int:
        return int(val * (1 << 16)) & 0xFFFFFFFF
        
    def q18_15_to_float(val: int) -> float:
        return val / (1 << 15)

    print("\n--- Test 1: Warmup Phase (Ticks 1-19) ---")
    for i in range(1, 20):
        bp = float_to_q17_15(60000.0)
        ap = float_to_q17_15(60000.0)
        bq = float_to_q16_16(1.0)
        aq = float_to_q16_16(0.0) # Total volume = 1.0
        
        valid, _ = vwap_engine.update(bp, ap, bq, aq)
        assert not valid, f"vwap_valid should be False on tick {i}"
        
    print("Warmup behavior verified (valid=False for first 19 ticks).")

    print("\n--- Test 2: First Valid Output (Tick 20) ---")
    # Add the 20th tick. The accumulators will have sum_pv = 20 * (60000 * 1.0)
    # sum_v = 20 * 1.0 = 20.0
    # VWAP should be exactly 60000.0
    bp = float_to_q17_15(60000.0)
    ap = float_to_q17_15(60000.0)
    bq = float_to_q16_16(1.0)
    aq = float_to_q16_16(0.0)
    
    valid, vwap = vwap_engine.update(bp, ap, bq, aq)
    assert valid, "vwap_valid should be True on tick 20"
    
    vwap_float = q18_15_to_float(vwap)
    print(f"VWAP = {vwap_float:.2f} (Expected: 60000.00)")
    assert abs(vwap_float - 60000.0) < 0.01

    print("\n--- Test 3: Eviction Correctness (Tick 21-40) ---")
    # Feed 20 ticks of price = 70000.0, volume = 2.0
    # This should completely evict the old 60000.0 ticks
    for i in range(21, 41):
        bp = float_to_q17_15(70000.0)
        ap = float_to_q17_15(70000.0)
        bq = float_to_q16_16(2.0)
        aq = float_to_q16_16(0.0)
        
        valid, vwap = vwap_engine.update(bp, ap, bq, aq)
        assert valid
        
    vwap_float = q18_15_to_float(vwap)
    print(f"VWAP at tick 40 = {vwap_float:.2f} (Expected: 70000.00)")
    assert abs(vwap_float - 70000.0) < 0.01

    print("\n--- Test 4: Exact Divisibility Boundary Case ---")
    # Clear the engine and set up an exact divisibility test
    exact_engine = GoldenVWAP(window_size=2)
    # Tick 1
    bp = float_to_q17_15(50000.0)
    ap = float_to_q17_15(50000.0)
    bq = float_to_q16_16(1.0)  # volume = 1.0
    aq = float_to_q16_16(0.0)
    exact_engine.update(bp, ap, bq, aq)
    
    # Tick 2
    bp = float_to_q17_15(60000.0)
    ap = float_to_q17_15(60000.0)
    bq = float_to_q16_16(1.0)  # volume = 1.0
    aq = float_to_q16_16(0.0)
    valid, vwap = exact_engine.update(bp, ap, bq, aq)
    
    # sum_pv = 50000*1 + 60000*1 = 110000
    # sum_v = 2
    # vwap should be exactly 55000.00 with zero remainder
    vwap_float = q18_15_to_float(vwap)
    print(f"Exact divisibility VWAP = {vwap_float:.2f} (Expected: 55000.00)")
    
    # The integer division must have 0 remainder
    # sum_pv is Q38.31, sum_v is Q20.16
    assert exact_engine.sum_pv % exact_engine.sum_v == 0, "Remainder should be exactly 0!"
    assert abs(vwap_float - 55000.0) < 0.01

    print("\n--- Test 5: Zero Volume Guard ---")
    zero_engine = GoldenVWAP(window_size=2)
    bp = float_to_q17_15(50000.0)
    ap = float_to_q17_15(50000.0)
    bq = float_to_q16_16(0.0)
    aq = float_to_q16_16(0.0)
    zero_engine.update(bp, ap, bq, aq)
    valid, vwap = zero_engine.update(bp, ap, bq, aq)
    
    print(f"Zero volume valid: {valid}, VWAP output: {vwap}")
    assert valid, "Valid should be true even on zero-division guard"
    assert vwap == 0, "VWAP must be exactly 0 on division by zero"

if __name__ == "__main__":
    test_manual_ticks()
    print("\nGolden VWAP sanity check complete. All assertions passed!")
