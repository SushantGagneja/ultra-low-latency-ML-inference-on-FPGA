#include "temporal_features.h"

#include <math.h>
#include <string.h>

static float select_price(const bnn_market_tick_t *tick)
{
    if (tick->bid > 0.0f && tick->ask > 0.0f && tick->ask >= tick->bid) {
        return 0.5f * (tick->bid + tick->ask);
    }
    return tick->price;
}

void bnn_feature_state_init(bnn_feature_state_t *s)
{
    memset(s, 0, sizeof(*s));
}

bool bnn_feature_update(bnn_feature_state_t *s,
                        const bnn_market_tick_t *tick,
                        bnn_indicators_t *out)
{
    const float price = select_price(tick);
    if (price <= 0.0f || tick->volume < 0.0f) {
        return false;
    }

    const uint32_t idx = s->idx;
    const uint32_t rsi_idx = s->rsi_idx;
    const bool window_full = s->count >= BNN_FEATURE_WINDOW;
    const bool rsi_full = s->count >= BNN_RSI_PERIOD;

    const float old_price = s->prices[idx];
    const float old_volume = s->volumes[idx];
    const float old_return = s->returns[idx];
    const float prev_price = (s->count == 0u) ? price : s->prev_price;

    const float delta = price - prev_price;
    const float gain = delta > 0.0f ? delta : 0.0f;
    const float loss = delta < 0.0f ? -delta : 0.0f;
    const float old_gain = s->gains[rsi_idx];
    const float old_loss = s->losses[rsi_idx];
    const float ret = (prev_price > 0.0f && s->count > 0u) ? ((price - prev_price) / prev_price) : 0.0f;

    if (window_full) {
        s->sum_price -= old_price;
        s->sum_volume -= old_volume;
        s->sum_return -= old_return;
        s->sum_return_sq -= old_return * old_return;
    }
    if (rsi_full) {
        s->sum_gain -= old_gain;
        s->sum_loss -= old_loss;
    }

    s->prices[idx] = price;
    s->volumes[idx] = tick->volume;
    s->returns[idx] = ret;
    s->gains[rsi_idx] = gain;
    s->losses[rsi_idx] = loss;

    s->sum_price += price;
    s->sum_volume += tick->volume;
    s->sum_return += ret;
    s->sum_return_sq += ret * ret;
    s->sum_gain += gain;
    s->sum_loss += loss;

    s->idx = (idx + 1u) & (BNN_FEATURE_WINDOW - 1u);
    s->rsi_idx = (rsi_idx + 1u) % BNN_RSI_PERIOD;
    if (s->count < BNN_FEATURE_WINDOW) {
        s->count++;
    }

    const uint32_t n = s->count;
    const float avg_volume = s->sum_volume / (float)n;
    const float volume_ratio = avg_volume > 0.0f ? tick->volume / avg_volume : 1.0f;
    const float mean_return = s->sum_return / (float)n;
    float variance = (s->sum_return_sq / (float)n) - (mean_return * mean_return);
    if (variance < 0.0f) {
        variance = 0.0f;
    }

    const float avg_gain = s->sum_gain / (float)((n < BNN_RSI_PERIOD) ? n : BNN_RSI_PERIOD);
    const float avg_loss = s->sum_loss / (float)((n < BNN_RSI_PERIOD) ? n : BNN_RSI_PERIOD);
    float rsi = 50.0f;
    if (avg_loss == 0.0f && avg_gain > 0.0f) {
        rsi = 100.0f;
    } else if (avg_loss > 0.0f) {
        const float rs = avg_gain / avg_loss;
        rsi = 100.0f - (100.0f / (1.0f + rs));
    }

    out->rsi = rsi;
    out->momentum = ret;
    out->volume_ratio = volume_ratio;
    out->volatility = sqrtf(variance);
    out->price = price;
    out->volume = tick->volume;
    out->prev_price = prev_price;

    s->prev_price = price;
    s->ready = n >= BNN_RSI_PERIOD;
    return s->ready;
}
