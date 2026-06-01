#!/usr/bin/env python3
"""
BNN Retraining with bookTicker Volume Contract
================================================

This script is a minimal fork of `train_bnn_standalone.py` that closes the
data-contract gap between Phase 1 (synthetic traded volume) and Phase 2
(bookTicker bid_qty + ask_qty as volume proxy).

Key change:
  The synthetic data generator now produces volume values that match the
  distribution of `bid_qty + ask_qty` from the Binance bookTicker stream
  for BTCUSDT, rather than the synthetic ~600-1200 range used previously.

  Typical bookTicker bid_qty + ask_qty for BTCUSDT:
    - Range: ~0.02 to ~80 BTC
    - Median: ~5 BTC
    - Distribution: heavy right tail (occasional large book updates)
    - Tick-to-tick variance: much higher than trade volume

After retraining:
  1. New weights are exported to fpga_weights/
  2. generate_test_vectors.py must be re-run
  3. `make` must pass 100/100
  4. cosim.py should confirm market-derived vectors still match
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import shutil
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Import the model architecture and utilities from Phase 1
# ---------------------------------------------------------------------------
# We add the parent directory to sys.path so we can import from the
# training script directly.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# We need the custom layers and activation, so we import them.
# The spaces in the filename require importlib.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_bnn", ROOT / "train_bnn_standalone.py"
)
train_bnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_bnn)

sign_with_ste   = train_bnn.sign_with_ste
BinaryDense     = train_bnn.BinaryDense
BinaryOutputDense = train_bnn.BinaryOutputDense
BipolarQuantizer = train_bnn.BipolarQuantizer
build_model     = train_bnn.build_model
train_model     = train_bnn.train_model
extract_weights = train_bnn.extract_weights
verify_fpga     = train_bnn.verify_fpga
fpga_inference_sim = train_bnn.fpga_inference_sim


# ---------------------------------------------------------------------------
# BNNFeatureExtractor: unified pipeline for scripts that need
# tick → indicators → quantized spike in a single .update() call.
#
# Used by:
#   - scripts/verify_c_equivalence.py
#   - scripts/historical_backtest.py
# ---------------------------------------------------------------------------

import math
from collections import deque


class BNNFeatureExtractor:
    """
    Wraps the rolling feature computation (RSI, momentum, volatility,
    volume ratio) and bipolar quantization into a single stateful object.

    Usage:
        extractor = BNNFeatureExtractor()
        ready, indicators_dict, spike_uint16 = extractor.update(tick_dict)

    tick_dict must have: 'price', 'volume', 'bid', 'ask'
    indicators_dict keys: 'rsi', 'momentum', 'volume_ratio', 'volatility',
                          'price', 'volume', 'prev_price'
    spike_uint16: 16-bit unsigned int bitmask ({0,1} per bit).

    Encoding note:
        The spike bitmask uses {0, 1} encoding (bit set = active feature).
        This is equivalent to the training pipeline's bipolar {-1, +1}
        encoding under XNOR-popcount. See quantization.c for proof.
    """

    def __init__(self, window: int = 64, rsi_period: int = 14):
        self.window = window
        self.rsi_period = rsi_period
        self.prices: deque = deque(maxlen=window)
        self.volumes: deque = deque(maxlen=window)
        self.returns: deque = deque(maxlen=window)
        self.gains: deque = deque(maxlen=rsi_period)
        self.losses: deque = deque(maxlen=rsi_period)
        self.prev_price = None
        self.prev_ind = None

        # Quantizer thresholds (must match quantization.c exactly)
        self.rsi_high = 70.0
        self.rsi_low = 30.0
        self.momentum_thr = 0.001
        self.momentum_strong = 0.005
        self.volume_ratio_high = 1.5
        self.volatility_high = 0.02
        self.volatility_extreme = 0.05

    def update(self, tick: dict):
        """
        Process one market tick.

        Returns:
            (ready, indicators_dict_or_None, spike_int_or_0)
        """
        bid = tick.get('bid', 0.0)
        ask = tick.get('ask', 0.0)
        if bid > 0 and ask >= bid:
            price = 0.5 * (bid + ask)
        else:
            price = tick['price']

        volume = tick['volume']
        if price <= 0 or volume < 0:
            return False, None, 0

        prev = price if self.prev_price is None else self.prev_price
        delta = price - prev
        ret = ((price - prev) / prev) if self.prev_price is not None and prev > 0 else 0.0

        self.prices.append(price)
        self.volumes.append(volume)
        self.returns.append(ret)
        self.gains.append(max(delta, 0.0))
        self.losses.append(max(-delta, 0.0))
        self.prev_price = price

        if len(self.prices) < self.rsi_period:
            return False, None, 0

        n = len(self.prices)
        avg_vol = sum(self.volumes) / n
        volume_ratio = volume / avg_vol if avg_vol > 0 else 1.0

        mean_ret = sum(self.returns) / n
        var = sum((r - mean_ret) ** 2 for r in self.returns) / n
        volatility = math.sqrt(max(var, 0.0))

        avg_gain = sum(self.gains) / len(self.gains)
        avg_loss = sum(self.losses) / len(self.losses)
        if avg_loss == 0.0 and avg_gain > 0.0:
            rsi = 100.0
        elif avg_loss > 0.0:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi = 50.0

        ind = {
            'rsi': rsi,
            'momentum': ret,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'price': price,
            'volume': volume,
            'prev_price': prev,
        }

        spike = self._quantize(ind)
        self.prev_ind = ind
        return True, ind, spike

    def _quantize(self, ind: dict) -> int:
        """Quantize indicators to a 16-bit bitmask. Mirrors quantization.c."""
        spike = 0

        if ind['rsi'] > self.rsi_high:
            spike |= 1 << 0
        if ind['rsi'] < self.rsi_low:
            spike |= 1 << 1

        mom = ind['momentum']
        if abs(mom) > self.momentum_thr:
            if mom > 0:
                spike |= 1 << 2
            if abs(mom) > self.momentum_strong:
                spike |= 1 << 3

        if ind['volume_ratio'] > self.volume_ratio_high:
            spike |= 1 << 4
        if ind['volume_ratio'] > 2.0:
            spike |= 1 << 5
        if ind['volatility'] > self.volatility_high:
            spike |= 1 << 6
        if ind['volatility'] > self.volatility_extreme:
            spike |= 1 << 7

        if self.prev_ind is not None:
            rsi_delta = ind['rsi'] - self.prev_ind['rsi']
            if abs(rsi_delta) > 1.0:
                if rsi_delta > 0:
                    spike |= 1 << 8
                if abs(rsi_delta) > 5.0:
                    spike |= 1 << 9

            pp = self.prev_ind['price']
            prev_pp = self.prev_ind.get('prev_price', pp)
            accel = (ind['price'] - pp) - (pp - prev_pp)
            if abs(accel) > 10.0:
                if accel > 0:
                    spike |= 1 << 10
                if abs(accel) > 100.0:
                    spike |= 1 << 11

            if self.prev_ind['volume'] > 0:
                vd = (ind['volume'] - self.prev_ind['volume']) / self.prev_ind['volume']
                if abs(vd) > 0.3:
                    if vd > 0:
                        spike |= 1 << 12
                    if abs(vd) > 0.7:
                        spike |= 1 << 13

            vol_delta = ind['volatility'] - self.prev_ind['volatility']
            if abs(vol_delta) > 0.01:
                if vol_delta > 0:
                    spike |= 1 << 14
                if abs(vol_delta) > 0.03:
                    spike |= 1 << 15

        return spike


# ---------------------------------------------------------------------------
# bookTicker-calibrated synthetic data generator
# ---------------------------------------------------------------------------

def generate_bookticker_data(n_samples: int = 12000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic BTC-style tick data with volume modeled after the
    Binance bookTicker bid_qty + ask_qty distribution.

    Volume model:
      bid_qty ~ LogNormal(mu=1.0, sigma=1.2)  → median ~2.7 BTC
      ask_qty ~ LogNormal(mu=1.0, sigma=1.2)  → median ~2.7 BTC
      volume  = bid_qty + ask_qty             → median ~5.4 BTC
      Range: ~0.02 to ~150 BTC (heavy right tail)

    This matches the empirical distribution of top-of-book liquidity on
    BTCUSDT far better than the previous uniform ~600-1200 range.
    """
    print(f"Generating {n_samples} bookTicker-calibrated samples...")
    quantizer = BipolarQuantizer()
    X, y = [], []

    price    = 50_000.0
    prev_ind = None
    buy_c = sell_c = hold_c = 0

    for i in range(n_samples):
        mom  = np.random.randn() * 0.008
        price = max(100.0, price * (1 + mom))

        rsi  = 50 + 40 * np.sin(i / 80.0) + np.random.randn() * 8
        rsi  = float(np.clip(rsi, 1, 99))

        # bookTicker-calibrated volume: LogNormal produces realistic
        # top-of-book liquidity with a heavy right tail
        bid_qty = float(np.random.lognormal(mean=1.0, sigma=1.2))
        ask_qty = float(np.random.lognormal(mean=1.0, sigma=1.2))
        volume  = bid_qty + ask_qty  # This is what the ESP32 computes

        # Volume ratio uses the same rolling-window logic as temporal_features.c
        # For training, we approximate with a simple ratio to a typical value
        vrat = volume / 5.4  # 5.4 ≈ median of bid_qty + ask_qty

        volt = max(0.005, 0.025 + 0.015 * abs(np.random.randn()))

        ind = {
            'rsi': rsi, 'momentum': mom,
            'volume_ratio': vrat, 'volatility': volt,
            'price': price, 'volume': volume,
            'prev_price': prev_ind['price'] if prev_ind else price
        }

        spike = quantizer.quantize(ind, prev_ind)
        X.append(spike)

        # --- Deterministic, balanced labels ---
        if rsi > 70 and mom > 0.004:
            label = [0, 0, 1];  sell_c += 1          # SELL
        elif rsi < 30 and mom < -0.004:
            label = [1, 0, 0];  buy_c  += 1          # BUY
        else:
            label = [0, 1, 0];  hold_c += 1          # HOLD

        y.append(label)
        prev_ind = {**ind}

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"  Samples : {n_samples}")
    print(f"  BUY     : {buy_c}  ({100*buy_c/n_samples:.1f}%)")
    print(f"  HOLD    : {hold_c} ({100*hold_c/n_samples:.1f}%)")
    print(f"  SELL    : {sell_c} ({100*sell_c/n_samples:.1f}%)")
    print(f"  Volume range: [{X[:, 4].min():.2f}, {X[:, 4].max():.2f}]  (bit 4 activations)")
    return X, y


def main():
    print("=" * 60)
    print("BNN RETRAINING — bookTicker Volume Contract Calibration")
    print("=" * 60)

    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Data (bookTicker-calibrated) ---
    X, y = generate_bookticker_data(12000)

    n      = len(X)
    tr_end = int(0.70 * n)
    val_end = int(0.85 * n)

    X_train, y_train = X[:tr_end],       y[:tr_end]
    X_val,   y_val   = X[tr_end:val_end], y[tr_end:val_end]
    X_test,  y_test  = X[val_end:],       y[val_end:]

    print(f"\n  Train : {len(X_train)}  Val : {len(X_val)}  Test : {len(X_test)}")

    # --- Model (same architecture) ---
    model   = build_model()
    history = train_model(model, X_train, y_train, X_val, y_val)

    # --- Evaluate ---
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'=' * 60}")
    print(f"  FINAL TEST ACCURACY : {acc*100:.2f}%")
    print(f"  FINAL TEST LOSS     : {loss:.4f}")
    print(f"{'=' * 60}")

    # --- Backup old weights ---
    weights_dir = ROOT / "fpga_weights"
    backup_dir  = ROOT / "fpga_weights_phase1_backup"
    if weights_dir.exists() and not backup_dir.exists():
        shutil.copytree(weights_dir, backup_dir)
        print(f"\n  Backed up original weights to {backup_dir}")

    # --- Extract new weights ---
    w1_bin, w2_bin = extract_weights(model, weights_dir)

    # Also copy weights.mem to project root (where BRAM init expects it)
    shutil.copy2(weights_dir / "weights.mem", ROOT / "weights.mem")

    # --- Verify hardware equivalence ---
    match_rate = verify_fpga(model, X_test, w1_bin, w2_bin)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("RETRAINING SUMMARY (bookTicker Volume Contract)")
    print(f"{'=' * 60}")
    print(f"  Test accuracy    : {acc*100:.2f}%")
    print(f"  HW match rate    : {match_rate:.1f}%")
    print(f"  Total parameters : {16*64 + 64*3} bits  ({(16*64+64*3)//8} bytes)")
    print(f"  BRAM usage       : {(16*64+64*3)/1024:.3f} kbits / 32 kbits")

    if match_rate >= 99.0 and acc >= 0.70:
        print("\n  ✅ RETRAINING COMPLETE — Weights calibrated for bookTicker contract")
        print("  Next steps:")
        print("    1. python scripts/generate_test_vectors.py")
        print("    2. make")
        print("    3. python scripts/cosim.py --vectors 500")
    else:
        print("\n  ⚠  Retraining incomplete — check warnings above")

    print("=" * 60)


if __name__ == "__main__":
    main()
extract_weights = train_bnn.extract_weights
verify_fpga     = train_bnn.verify_fpga
fpga_inference_sim = train_bnn.fpga_inference_sim


# ---------------------------------------------------------------------------
# bookTicker-calibrated synthetic data generator
# ---------------------------------------------------------------------------

def generate_bookticker_data(n_samples: int = 12000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic BTC-style tick data with volume modeled after the
    Binance bookTicker bid_qty + ask_qty distribution.

    Volume model:
      bid_qty ~ LogNormal(mu=1.0, sigma=1.2)  → median ~2.7 BTC
      ask_qty ~ LogNormal(mu=1.0, sigma=1.2)  → median ~2.7 BTC
      volume  = bid_qty + ask_qty             → median ~5.4 BTC
      Range: ~0.02 to ~150 BTC (heavy right tail)

    This matches the empirical distribution of top-of-book liquidity on
    BTCUSDT far better than the previous uniform ~600-1200 range.
    """
    print(f"Generating {n_samples} bookTicker-calibrated samples...")
    quantizer = BipolarQuantizer()
    X, y = [], []

    price    = 50_000.0
    prev_ind = None
    buy_c = sell_c = hold_c = 0

    for i in range(n_samples):
        mom  = np.random.randn() * 0.008
        price = max(100.0, price * (1 + mom))

        rsi  = 50 + 40 * np.sin(i / 80.0) + np.random.randn() * 8
        rsi  = float(np.clip(rsi, 1, 99))

        # bookTicker-calibrated volume: LogNormal produces realistic
        # top-of-book liquidity with a heavy right tail
        bid_qty = float(np.random.lognormal(mean=1.0, sigma=1.2))
        ask_qty = float(np.random.lognormal(mean=1.0, sigma=1.2))
        volume  = bid_qty + ask_qty  # This is what the ESP32 computes

        # Volume ratio uses the same rolling-window logic as temporal_features.c
        # For training, we approximate with a simple ratio to a typical value
        vrat = volume / 5.4  # 5.4 ≈ median of bid_qty + ask_qty

        volt = max(0.005, 0.025 + 0.015 * abs(np.random.randn()))

        ind = {
            'rsi': rsi, 'momentum': mom,
            'volume_ratio': vrat, 'volatility': volt,
            'price': price, 'volume': volume,
            'prev_price': prev_ind['price'] if prev_ind else price
        }

        spike = quantizer.quantize(ind, prev_ind)
        X.append(spike)

        # --- Deterministic, balanced labels ---
        if rsi > 70 and mom > 0.004:
            label = [0, 0, 1];  sell_c += 1          # SELL
        elif rsi < 30 and mom < -0.004:
            label = [1, 0, 0];  buy_c  += 1          # BUY
        else:
            label = [0, 1, 0];  hold_c += 1          # HOLD

        y.append(label)
        prev_ind = {**ind}

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"  Samples : {n_samples}")
    print(f"  BUY     : {buy_c}  ({100*buy_c/n_samples:.1f}%)")
    print(f"  HOLD    : {hold_c} ({100*hold_c/n_samples:.1f}%)")
    print(f"  SELL    : {sell_c} ({100*sell_c/n_samples:.1f}%)")
    print(f"  Volume range: [{X[:, 4].min():.2f}, {X[:, 4].max():.2f}]  (bit 4 activations)")
    return X, y


def main():
    print("=" * 60)
    print("BNN RETRAINING — bookTicker Volume Contract Calibration")
    print("=" * 60)

    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Data (bookTicker-calibrated) ---
    X, y = generate_bookticker_data(12000)

    n      = len(X)
    tr_end = int(0.70 * n)
    val_end = int(0.85 * n)

    X_train, y_train = X[:tr_end],       y[:tr_end]
    X_val,   y_val   = X[tr_end:val_end], y[tr_end:val_end]
    X_test,  y_test  = X[val_end:],       y[val_end:]

    print(f"\n  Train : {len(X_train)}  Val : {len(X_val)}  Test : {len(X_test)}")

    # --- Model (same architecture) ---
    model   = build_model()
    history = train_model(model, X_train, y_train, X_val, y_val)

    # --- Evaluate ---
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'=' * 60}")
    print(f"  FINAL TEST ACCURACY : {acc*100:.2f}%")
    print(f"  FINAL TEST LOSS     : {loss:.4f}")
    print(f"{'=' * 60}")

    # --- Backup old weights ---
    weights_dir = ROOT / "fpga_weights"
    backup_dir  = ROOT / "fpga_weights_phase1_backup"
    if weights_dir.exists() and not backup_dir.exists():
        shutil.copytree(weights_dir, backup_dir)
        print(f"\n  Backed up original weights to {backup_dir}")

    # --- Extract new weights ---
    w1_bin, w2_bin = extract_weights(model, weights_dir)

    # Also copy weights.mem to project root (where BRAM init expects it)
    shutil.copy2(weights_dir / "weights.mem", ROOT / "weights.mem")

    # --- Verify hardware equivalence ---
    match_rate = verify_fpga(model, X_test, w1_bin, w2_bin)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("RETRAINING SUMMARY (bookTicker Volume Contract)")
    print(f"{'=' * 60}")
    print(f"  Test accuracy    : {acc*100:.2f}%")
    print(f"  HW match rate    : {match_rate:.1f}%")
    print(f"  Total parameters : {16*64 + 64*3} bits  ({(16*64+64*3)//8} bytes)")
    print(f"  BRAM usage       : {(16*64+64*3)/1024:.3f} kbits / 32 kbits")

    if match_rate >= 99.0 and acc >= 0.70:
        print("\n  ✅ RETRAINING COMPLETE — Weights calibrated for bookTicker contract")
        print("  Next steps:")
        print("    1. python scripts/generate_test_vectors.py")
        print("    2. make")
        print("    3. python scripts/cosim.py --vectors 500")
    else:
        print("\n  ⚠  Retraining incomplete — check warnings above")

    print("=" * 60)


if __name__ == "__main__":
    main()
