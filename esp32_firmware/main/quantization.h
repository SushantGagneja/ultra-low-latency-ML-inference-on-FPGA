#ifndef BNN_QUANTIZATION_H
#define BNN_QUANTIZATION_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    float rsi;
    float momentum;
    float volume_ratio;
    float volatility;
    float price;
    float volume;
    float prev_price;
} bnn_indicators_t;

typedef struct {
    float rsi_high;
    float rsi_low;
    float momentum_thr;
    float momentum_strong;
    float volume_ratio_high;
    float volatility_high;
    float volatility_extreme;
    bool has_prev;
    bnn_indicators_t prev;
} bnn_quantizer_t;

void bnn_quantizer_init(bnn_quantizer_t *q);
uint16_t bnn_quantize_bipolar(bnn_quantizer_t *q, const bnn_indicators_t *ind);

#endif
