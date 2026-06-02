#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

# Fixed-point formats
# Q17.15 for Price: 17 bits integer, 15 bits fractional. Total 32 bits.
# Q16.16 for Volume: 16 bits integer, 16 bits fractional. Total 32 bits.

def float_to_q17_15(val: float) -> int:
    """Convert float to Q17.15 unsigned integer, mimicking Verilog truncation."""
    q = int(val * (1 << 15))
    return q & 0xFFFFFFFF

def float_to_q16_16(val: float) -> int:
    """Convert float to Q16.16 unsigned integer, mimicking Verilog truncation."""
    q = int(val * (1 << 16))
    return q & 0xFFFFFFFF

def to_signed_32(val: int) -> int:
    """Convert a 32-bit unsigned integer to a signed Python int."""
    val = val & 0xFFFFFFFF
    if val >= 0x80000000:
        val -= 0x100000000
    return val

class GoldenOFI:
    def __init__(self):
        self.prev_bid_price = 0  # Q17.15
        self.prev_ask_price = 0  # Q17.15
        self.prev_bid_qty = 0    # Q16.16
        self.prev_ask_qty = 0    # Q16.16
        self.is_first_tick = True

    def compute_ofi(self, bid_price_q: int, ask_price_q: int, bid_qty_q: int, ask_qty_q: int) -> int:
        """
        Compute standard Cont-Kukanov-Stoikov OFI in pure Q16.16 integer arithmetic.
        
        True CKS formula for depth change:
        bid_side = bid_size * I(bid >= prev_bid) - prev_bid_size * I(bid <= prev_bid)
        ask_side = ask_size * I(ask <= prev_ask) - prev_ask_size * I(ask >= prev_ask)
        OFI = bid_side - ask_side
        """
        if self.is_first_tick:
            self.prev_bid_price = bid_price_q
            self.prev_ask_price = ask_price_q
            self.prev_bid_qty = bid_qty_q
            self.prev_ask_qty = ask_qty_q
            self.is_first_tick = False
            return 0
            
        # Indicator functions
        i_bid_ge = 1 if bid_price_q >= self.prev_bid_price else 0
        i_bid_le = 1 if bid_price_q <= self.prev_bid_price else 0
        i_ask_le = 1 if ask_price_q <= self.prev_ask_price else 0
        i_ask_ge = 1 if ask_price_q >= self.prev_ask_price else 0
        
        # Accumulate signed depth changes. Volumes are Q16.16.
        # We must maintain strict 32-bit 2's complement arithmetic.
        term1 = (bid_qty_q * i_bid_ge) & 0xFFFFFFFF
        term2 = (self.prev_bid_qty * i_bid_le) & 0xFFFFFFFF
        term3 = (ask_qty_q * i_ask_le) & 0xFFFFFFFF
        term4 = (self.prev_ask_qty * i_ask_ge) & 0xFFFFFFFF
        
        # OFI = term1 - term2 - term3 + term4
        ofi = term1
        ofi = (ofi - term2) & 0xFFFFFFFF
        ofi = (ofi - term3) & 0xFFFFFFFF
        ofi = (ofi + term4) & 0xFFFFFFFF
        
        # Update state
        self.prev_bid_price = bid_price_q
        self.prev_ask_price = ask_price_q
        self.prev_bid_qty = bid_qty_q
        self.prev_ask_qty = ask_qty_q
        
        return to_signed_32(ofi)

def test_manual_ticks():
    print("Testing Golden OFI with manual scenarios...")
    ofi_engine = GoldenOFI()
    
    # Helper to convert Q16.16 back to float for printing
    def q16_to_float(q: int) -> float:
        return q / (1 << 16)
        
    print("\n--- Tick 1: Baseline ---")
    bp = float_to_q17_15(60000.00); ap = float_to_q17_15(60000.50)
    bq = float_to_q16_16(1.0); aq = float_to_q16_16(1.0)
    res = ofi_engine.compute_ofi(bp, ap, bq, aq)
    print(f"OFI = {q16_to_float(res):.2f} (Expected: 0.00)")
    assert res == 0
    
    print("\n--- Tick 2: No price change, Bid depth +0.5 ---")
    bp = float_to_q17_15(60000.00); ap = float_to_q17_15(60000.50)
    bq = float_to_q16_16(1.5); aq = float_to_q16_16(1.0)
    res = ofi_engine.compute_ofi(bp, ap, bq, aq)
    print(f"OFI = {q16_to_float(res):.2f} (Expected: 0.50)")
    assert q16_to_float(res) == 0.50
    
    print("\n--- Tick 3: No price change, Ask depth +2.0 ---")
    bp = float_to_q17_15(60000.00); ap = float_to_q17_15(60000.50)
    bq = float_to_q16_16(1.5); aq = float_to_q16_16(3.0)
    res = ofi_engine.compute_ofi(bp, ap, bq, aq)
    print(f"OFI = {q16_to_float(res):.2f} (Expected: -2.00)")
    assert q16_to_float(res) == -2.00
    
    print("\n--- Tick 4: Bid price increases (old bid gone, new bid depth 0.8) ---")
    bp = float_to_q17_15(60000.10); ap = float_to_q17_15(60000.50)
    bq = float_to_q16_16(0.8); aq = float_to_q16_16(3.0)
    res = ofi_engine.compute_ofi(bp, ap, bq, aq)
    print(f"OFI = {q16_to_float(res):.2f} (Expected: 0.80)")
    assert round(q16_to_float(res), 2) == 0.80
    
    print("\n--- Tick 5: Ask price decreases (old ask gone, new ask depth 1.2) ---")
    bp = float_to_q17_15(60000.10); ap = float_to_q17_15(60000.20)
    bq = float_to_q16_16(0.8); aq = float_to_q16_16(1.2)
    res = ofi_engine.compute_ofi(bp, ap, bq, aq)
    print(f"OFI = {q16_to_float(res):.2f} (Expected: -1.20)")
    assert round(q16_to_float(res), 2) == -1.20

    print("\n--- Tick 6: Bid price drops (support broken, lost previous 0.8 depth) ---")
    bp = float_to_q17_15(60000.00); ap = float_to_q17_15(60000.20)
    bq = float_to_q16_16(5.0); aq = float_to_q16_16(1.2)
    res = ofi_engine.compute_ofi(bp, ap, bq, aq)
    print(f"OFI = {q16_to_float(res):.2f} (Expected: -0.80)")
    assert round(q16_to_float(res), 2) == -0.80

if __name__ == "__main__":
    test_manual_ticks()
    print("\nGolden OFI sanity check complete. All assertions passed!")
