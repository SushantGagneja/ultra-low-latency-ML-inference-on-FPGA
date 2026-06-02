#!/usr/bin/env python3

class GoldenQuantizer:
    def __init__(self):
        # Encodings
        self.LR_NEUTRAL = 0b00
        self.LR_BUYER   = 0b01
        self.LR_SELLER  = 0b10
        self.LR_UNDEF   = 0b11

    def update(self, ofi_q16_16: int, vwap_q18_15: int, midpoint: int, lr_class: int, 
               ofi_valid: bool, vwap_valid: bool, lr_valid: bool) -> tuple[bool, int]:
        
        feature_valid = ofi_valid and vwap_valid and lr_valid
        
        if not feature_valid:
            return False, 0
            
        # Align VWAP: Q18.15 -> Q17.15 by shifting right by 1
        # This mirrors vwap_q18_15[33:1] in RTL
        vwap_aligned = vwap_q18_15 >> 1
        
        # Midpoint is already Q17.15, zero extended conceptually to 33 bits in RTL
        midpoint_ext = midpoint
        
        spike_vector = 0
        
        # Bit 0: OFI > +327680 (strong buying pressure)
        if ofi_q16_16 > 327680:
            spike_vector |= (1 << 0)
            
        # Bit 1: OFI > 0 (any buying pressure)
        if ofi_q16_16 > 0:
            spike_vector |= (1 << 1)
            
        # Bit 2: OFI < -327680 (strong selling pressure)
        if ofi_q16_16 < -327680:
            spike_vector |= (1 << 2)
            
        # Bit 3: OFI < 0 (any selling pressure)
        if ofi_q16_16 < 0:
            spike_vector |= (1 << 3)
            
        # Bit 4: midpoint > vwap
        if midpoint_ext > vwap_aligned:
            spike_vector |= (1 << 4)
            
        # Bit 5: midpoint < vwap
        if midpoint_ext < vwap_aligned:
            spike_vector |= (1 << 5)
            
        # Bit 6: lr_class == LR_BUYER
        if lr_class == self.LR_BUYER:
            spike_vector |= (1 << 6)
            
        # Bit 7: lr_class == LR_SELLER
        if lr_class == self.LR_SELLER:
            spike_vector |= (1 << 7)
            
        # Bit 8: feature pipeline ready
        spike_vector |= (1 << 8)
        
        # Bits 9-15: 0 (already 0)
        
        return True, spike_vector

def test_manual_scenarios():
    print("Testing Golden Quantizer with manual scenarios...")
    quantizer = GoldenQuantizer()
    
    print("\n--- Test 1: Exact OFI Threshold (Strict Inequality) ---")
    # ofi_q16_16 = 327680 exactly.
    # Bits 0 and 2 must be 0 (no strong pressure). Bit 1 must be 1 (ofi > 0).
    valid, sv = quantizer.update(
        ofi_q16_16=327680, 
        vwap_q18_15=0, 
        midpoint=0, 
        lr_class=0b00, 
        ofi_valid=True, vwap_valid=True, lr_valid=True
    )
    
    assert valid
    assert (sv & (1 << 0)) == 0, "Bit 0 should be 0 (strict inequality for strong buy)"
    assert (sv & (1 << 1)) != 0, "Bit 1 should be 1 (any buy)"
    assert (sv & (1 << 2)) == 0, "Bit 2 should be 0"
    print(f"OFI=+327680 -> SV Bit0={(sv>>0)&1}, Bit1={(sv>>1)&1}, Bit2={(sv>>2)&1}")
    
    print("\n--- Test 2: Negative OFI Threshold (Strict Inequality) ---")
    valid, sv = quantizer.update(
        ofi_q16_16=-327680, 
        vwap_q18_15=0, 
        midpoint=0, 
        lr_class=0b00, 
        ofi_valid=True, vwap_valid=True, lr_valid=True
    )
    
    assert valid
    assert (sv & (1 << 0)) == 0, "Bit 0 should be 0"
    assert (sv & (1 << 2)) == 0, "Bit 2 should be 0 (strict inequality for strong sell)"
    assert (sv & (1 << 3)) != 0, "Bit 3 should be 1 (any sell)"
    print(f"OFI=-327680 -> SV Bit0={(sv>>0)&1}, Bit2={(sv>>2)&1}, Bit3={(sv>>3)&1}")
    
    print("\n--- Test 3: VWAP Precision Alignment ---")
    # vwap is 34-bit Q18.15. Let's make it such that when shifted right by 1, it equals midpoint.
    # If midpoint = 50, and vwap = 101. 101 >> 1 = 50. midpoint_ext = 50.
    # They are equal, so neither bit 4 nor 5 should be set.
    valid, sv = quantizer.update(
        ofi_q16_16=0, 
        vwap_q18_15=101, 
        midpoint=50, 
        lr_class=0b00, 
        ofi_valid=True, vwap_valid=True, lr_valid=True
    )
    
    assert valid
    assert (sv & (1 << 4)) == 0, "Bit 4 should be 0 (midpoint == vwap_aligned)"
    assert (sv & (1 << 5)) == 0, "Bit 5 should be 0 (midpoint == vwap_aligned)"
    print(f"VWAP=101, Midpoint=50 -> SV Bit4={(sv>>4)&1}, Bit5={(sv>>5)&1}")
    
    # VWAP = 99 >> 1 = 49. Midpoint = 50. Midpoint > VWAP_aligned
    valid, sv = quantizer.update(
        ofi_q16_16=0, 
        vwap_q18_15=99, 
        midpoint=50, 
        lr_class=0b00, 
        ofi_valid=True, vwap_valid=True, lr_valid=True
    )
    assert valid
    assert (sv & (1 << 4)) != 0, "Bit 4 should be 1 (midpoint > vwap_aligned)"
    assert (sv & (1 << 5)) == 0
    print(f"VWAP=99, Midpoint=50 -> SV Bit4={(sv>>4)&1}, Bit5={(sv>>5)&1}")

if __name__ == "__main__":
    test_manual_scenarios()
    print("\nGolden Quantizer sanity check complete. All assertions passed!")
