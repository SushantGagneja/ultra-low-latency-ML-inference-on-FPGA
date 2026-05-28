#ifndef BNN_TEMPORAL_FEATURES_H
#define BNN_TEMPORAL_FEATURES_H

#include <stdbool.h>
#include <stdint.h>

#include "quantization.h"

#define BNN_FEATURE_WINDOW 64u
#define BNN_RSI_PERIOD 14u

typedef struct {
    float price;
    float bid;
    float ask;
    float volume;
    int64_t timestamp_us;
} bnn_market_tick_t;

typedef struct {
    bool ready;
    uint32_t count;
    uint32_t idx;
    uint32_t rsi_idx;
    float prev_price;
    float prices[BNN_FEATURE_WINDOW];
    float volumes[BNN_FEATURE_WINDOW];
    float returns[BNN_FEATURE_WINDOW];
    float gains[BNN_RSI_PERIOD];
    float losses[BNN_RSI_PERIOD];
    float sum_price;
    float sum_volume;
    float sum_return;
    float sum_return_sq;
    float sum_gain;
    float sum_loss;
} bnn_feature_state_t;

void bnn_feature_state_init(bnn_feature_state_t *s);
bool bnn_feature_update(bnn_feature_state_t *s,
                        const bnn_market_tick_t *tick,
                        bnn_indicators_t *out);

#endif
