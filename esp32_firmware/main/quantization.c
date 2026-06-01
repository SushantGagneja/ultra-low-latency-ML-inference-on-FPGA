#include "quantization.h"

#include <math.h>
#include <string.h>

void bnn_quantizer_init(bnn_quantizer_t *q)
{
    memset(q, 0, sizeof(*q));
    q->rsi_high = 70.0f;
    q->rsi_low = 30.0f;
    q->momentum_thr = 0.001f;
    q->momentum_strong = 0.005f;
    q->volume_ratio_high = 1.5f;
    q->volatility_high = 0.02f;
    q->volatility_extreme = 0.05f;
}

uint16_t bnn_quantize_bipolar(bnn_quantizer_t *q, const bnn_indicators_t *ind)
{
    // -----------------------------------------------------------------------
    // ENCODING EQUIVALENCE NOTE:
    //
    // This function produces a uint16_t bitmask in the {0, 1} domain:
    //   bit=1  →  feature is active (e.g., RSI > 70)
    //   bit=0  →  feature is inactive
    //
    // The Python training pipeline (train_bnn_standalone.py) uses bipolar
    // {-1, +1} encoding:
    //   +1 → feature active,  -1 → feature inactive
    //
    // These two representations are equivalent under XNOR-popcount:
    //   XNOR(0, 0) = 1  ↔  (-1) × (-1) = +1  (agreement)
    //   XNOR(1, 1) = 1  ↔  (+1) × (+1) = +1  (agreement)
    //   XNOR(0, 1) = 0  ↔  (-1) × (+1) = -1  (disagreement)
    //   XNOR(1, 0) = 0  ↔  (+1) × (-1) = -1  (disagreement)
    //
    // The conversion between domains is: bit = (bipolar + 1) / 2
    // The FPGA's XNOR gates and Python's bipolar matmul produce identical
    // popcount results, so no runtime conversion is needed.
    // -----------------------------------------------------------------------
    uint16_t spike = 0u;

    if (ind->rsi > q->rsi_high) {
        spike |= (uint16_t)(1u << 0);
    }

    if (ind->rsi < q->rsi_low) {
        spike |= (uint16_t)(1u << 1);
    }

    if (fabsf(ind->momentum) > q->momentum_thr) {
        if (ind->momentum > 0.0f) {
            spike |= (uint16_t)(1u << 2);
        }
        if (fabsf(ind->momentum) > q->momentum_strong) {
            spike |= (uint16_t)(1u << 3);
        }
    }

    if (ind->volume_ratio > q->volume_ratio_high) {
        spike |= (uint16_t)(1u << 4);
    }
    if (ind->volume_ratio > 2.0f) {
        spike |= (uint16_t)(1u << 5);
    }
    if (ind->volatility > q->volatility_high) {
        spike |= (uint16_t)(1u << 6);
    }
    if (ind->volatility > q->volatility_extreme) {
        spike |= (uint16_t)(1u << 7);
    }

    if (q->has_prev) {
        const float rsi_delta = ind->rsi - q->prev.rsi;
        if (fabsf(rsi_delta) > 1.0f) {
            if (rsi_delta > 0.0f) {
                spike |= (uint16_t)(1u << 8);
            }
            if (fabsf(rsi_delta) > 5.0f) {
                spike |= (uint16_t)(1u << 9);
            }
        }

        const float price_accel =
            (ind->price - q->prev.price) - (q->prev.price - q->prev.prev_price);
        if (fabsf(price_accel) > 10.0f) {
            if (price_accel > 0.0f) {
                spike |= (uint16_t)(1u << 10);
            }
            if (fabsf(price_accel) > 100.0f) {
                spike |= (uint16_t)(1u << 11);
            }
        }

        if (q->prev.volume > 0.0f) {
            const float volume_delta_ratio = (ind->volume - q->prev.volume) / q->prev.volume;
            if (fabsf(volume_delta_ratio) > 0.3f) {
                if (volume_delta_ratio > 0.0f) {
                    spike |= (uint16_t)(1u << 12);
                }
                if (fabsf(volume_delta_ratio) > 0.7f) {
                    spike |= (uint16_t)(1u << 13);
                }
            }
        }

        const float vol_delta = ind->volatility - q->prev.volatility;
        if (fabsf(vol_delta) > 0.01f) {
            if (vol_delta > 0.0f) {
                spike |= (uint16_t)(1u << 14);
            }
            if (fabsf(vol_delta) > 0.03f) {
                spike |= (uint16_t)(1u << 15);
            }
        }
    }

    q->prev = *ind;
    q->has_prev = true;
    return spike;
}
