#!/usr/bin/env python3
import sys

# Output Encodings
LR_NEUTRAL = 0b00
LR_BUYER   = 0b01
LR_SELLER  = 0b10
LR_UNDEF   = 0b11

def to_unsigned(val: int, bits: int) -> int:
    mask = (1 << bits) - 1
    return val & mask

class GoldenLeeReady:
    def __init__(self):
        self.midpoint_prev = None
        self.tick_count = 0

    def update(self, bid_price_q: int, ask_price_q: int):
        """
        Input:
        - bid_price_q, ask_price_q (Q17.15): 32 bits
        
        Returns:
        - lr_valid (bool)
        - lr_class (int, 2 bits)
        """
        self.tick_count += 1
        
        # Calculate current midpoint
        # price = (bid + ask) >> 1
        midpoint_curr = to_unsigned((bid_price_q + ask_price_q) >> 1, 32)
        
        if self.midpoint_prev is None:
            # First tick: no previous midpoint to compare against
            classification = LR_UNDEF
            lr_valid = False
        else:
            # Subsequent ticks: compare against previous midpoint
            if midpoint_curr > self.midpoint_prev:
                classification = LR_BUYER
            elif midpoint_curr < self.midpoint_prev:
                classification = LR_SELLER
            else:
                classification = LR_NEUTRAL
            lr_valid = True
            
        # Register for next tick
        self.midpoint_prev = midpoint_curr
        
        return lr_valid, classification

def test_manual_ticks():
    print("Testing Golden Lee-Ready with manual scenarios...")
    lr_engine = GoldenLeeReady()
    
    # Helper to convert float to fixed-point
    def float_to_q17_15(val: float) -> int:
        return int(val * (1 << 15)) & 0xFFFFFFFF

    print("\n--- Test 1: Three consecutive identical midpoints ---")
    
    # Tick 1: Midpoint = 60000.0 (Undefined output)
    bp = float_to_q17_15(60000.0)
    ap = float_to_q17_15(60000.0)
    valid, classification = lr_engine.update(bp, ap)
    print(f"Tick 1: valid={valid}, class={classification} (Expected: False, {LR_UNDEF})")
    assert not valid
    assert classification == LR_UNDEF
    
    # Tick 2: Midpoint = 60000.0 (Neutral output)
    valid, classification = lr_engine.update(bp, ap)
    print(f"Tick 2: valid={valid}, class={classification} (Expected: True, {LR_NEUTRAL})")
    assert valid
    assert classification == LR_NEUTRAL
    
    # Tick 3: Midpoint = 60000.0 (Neutral output)
    valid, classification = lr_engine.update(bp, ap)
    print(f"Tick 3: valid={valid}, class={classification} (Expected: True, {LR_NEUTRAL})")
    assert valid
    assert classification == LR_NEUTRAL

    print("\n--- Test 2: Midpoint increases by exactly 1 LSB ---")
    # LSB in Q17.15 is exactly 1 integer value after shift
    # If previous sum was S, we want new sum to be S + 2 (so after >> 1, it increases by 1)
    
    # Current midpoint fixed-point value
    prev_midpoint_q = to_unsigned((bp + ap) >> 1, 32)
    
    # Tick 4: Increase ask by 2 units in fixed point (meaning sum increases by 2, midpoint by 1)
    bp_new = bp
    ap_new = ap + 2
    
    new_midpoint_q = to_unsigned((bp_new + ap_new) >> 1, 32)
    print(f"Prev Midpoint Q: {prev_midpoint_q}, New Midpoint Q: {new_midpoint_q}")
    assert new_midpoint_q == prev_midpoint_q + 1, "Midpoint did not increase by exactly 1 LSB!"
    
    valid, classification = lr_engine.update(bp_new, ap_new)
    print(f"Tick 4: valid={valid}, class={classification} (Expected: True, {LR_BUYER})")
    assert valid
    assert classification == LR_BUYER

    print("\n--- Test 3: Midpoint decreases by exactly 1 LSB ---")
    # Tick 5: Decrease bid by 4 units (midpoint decreases by 2, which is 1 LSB below prev_prev)
    bp_new2 = bp_new - 4
    ap_new2 = ap_new
    
    new_midpoint_q2 = to_unsigned((bp_new2 + ap_new2) >> 1, 32)
    print(f"Prev Midpoint Q: {new_midpoint_q}, New Midpoint Q: {new_midpoint_q2}")
    assert new_midpoint_q2 == new_midpoint_q - 2, "Midpoint did not decrease by exactly 2 LSBs!"
    
    valid, classification = lr_engine.update(bp_new2, ap_new2)
    print(f"Tick 5: valid={valid}, class={classification} (Expected: True, {LR_SELLER})")
    assert valid
    assert classification == LR_SELLER

if __name__ == "__main__":
    test_manual_ticks()
    print("\nGolden Lee-Ready sanity check complete. All assertions passed!")
