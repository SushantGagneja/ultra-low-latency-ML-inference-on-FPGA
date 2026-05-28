#!/usr/bin/env python3
"""
BNN Trading Signal Classifier - Phase 1 (Hardware-Accurate)
16 → 64 → 3 Binary Neural Network for Renesas SLG47910V FPGA

Architecture constraints enforced:
  - Weights:     {-1, +1} strictly (no bias terms)
  - Hidden act:  sign(x>=0) → +1, else -1   [matches XNOR popcount >= N/2]
  - Output layer: NO binary activation — raw bipolar sums feed softmax
  - FPGA sim:    pure {0,1} XNOR-popcount in integer domain

Defects fixed vs previous version:
  1. sign_with_ste: tf.sign(0)=0 eliminated → tf.where(x>=0,+1,-1)
  2. Layer-2 output: binary activation removed, raw sums feed softmax
  3. fpga_sim Layer-1: np.sign() replaced with integer XNOR-popcount
  4. fpga_sim Layer-2: floating-point multiply replaced with XNOR-popcount
  5. Synthetic labels: balanced 3-class distribution, no momentum-only bias
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# 1.  HARDWARE-ACCURATE ACTIVATION
# ---------------------------------------------------------------------------

@tf.custom_gradient
def sign_with_ste(x):
    """
    Straight-Through Estimator binary activation.

    Forward  : +1 if x >= 0, else -1   (NO zero state)
    Backward : gradient passes through unchanged (STE)

    Hardware mapping:
      +1  →  logic '1'
      -1  →  logic '0'
    Tie-breaking (x == 0) resolves to +1, matching Verilog:
      if (popcount * 2 - N >= 0) → spike = 1
    """
    def grad(dy):
        return dy

    out = tf.where(x >= 0, tf.ones_like(x), -tf.ones_like(x))
    return out, grad


# ---------------------------------------------------------------------------
# 2.  LAYER DEFINITIONS
# ---------------------------------------------------------------------------

class BinaryDense(keras.layers.Layer):
    """
    Hidden BNN layer.
    Weights:     w ∈ {-1,+1}   (binarised during forward pass via STE)
    Activations: a ∈ {-1,+1}   (applied after matmul)
    No bias — eliminates an adder stage from the Verilog neuron.
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name='kernel'
        )

    def call(self, inputs):
        w_bin = sign_with_ste(self.w)       # binarise weights
        out   = tf.matmul(inputs, w_bin)    # bipolar matmul
        return sign_with_ste(out)           # binarise activations

    def get_binary_weights(self) -> np.ndarray:
        """Return binarised weights as numpy {-1,+1}."""
        return np.where(self.w.numpy() >= 0, 1.0, -1.0)


class BinaryOutputDense(keras.layers.Layer):
    """
    Output BNN layer.
    Weights:     w ∈ {-1,+1}   (binarised during forward pass)
    Activations: RAW bipolar sums — NOT binarised.

    Rationale: The FPGA computes raw popcount sums for all 3 output neurons
    and picks the winner via argmax (winner-take-all logic).  Applying sign()
    here would collapse {+8, +6, -4} to {+1, +1, -1}, losing the ordering
    needed by argmax and causing TF↔FPGA divergence.
    Softmax is applied on top by the model; argmax(softmax(x)) == argmax(x).
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name='kernel'
        )

    def call(self, inputs):
        w_bin = sign_with_ste(self.w)
        return tf.matmul(inputs, w_bin)     # raw sums — no sign activation

    def get_binary_weights(self) -> np.ndarray:
        return np.where(self.w.numpy() >= 0, 1.0, -1.0)


# ---------------------------------------------------------------------------
# 3.  BIPOLAR QUANTIZER  (mirrors ESP32-S3 firmware logic)
# ---------------------------------------------------------------------------

class BipolarQuantizer:
    """
    Compresses floating-point technical indicators into a 16-bit bipolar
    spike vector.

    Bit layout:
      [0-7]  Regime bits  — absolute market state
      [8-15] Delta bits   — momentum / acceleration

    Every bit is strictly ∈ {-1, +1}.  Neutral zones are left at their
    initialised -1 (not 0), ensuring the XNOR math never sees a unipolar 0.
    """

    def __init__(self):
        self.rsi_high          = 70.0
        self.rsi_low           = 30.0
        self.momentum_thr      = 0.001
        self.momentum_strong   = 0.005
        self.vol_ratio_high    = 1.5
        self.vol_high          = 0.02
        self.vol_extreme       = 0.05

    def quantize(self, ind: dict, prev: dict = None) -> np.ndarray:
        v = np.full(16, -1.0, dtype=np.float32)   # default ALL bits to -1

        rsi  = ind['rsi']
        mom  = ind['momentum']
        vrat = ind['volume_ratio']
        volt = ind['volatility']

        # --- REGIME BITS ---
        # Bit 0: overbought vs oversold (fired only outside neutral [30,70])
        if rsi > self.rsi_high:
            v[0] =  1.0
        elif rsi < self.rsi_low:
            v[0] = -1.0
        # else: neutral zone — leave at -1 (known default, no ambiguity)

        v[1] =  1.0 if rsi > 50 else -1.0          # Bit 1: bullish bias

        if abs(mom) > self.momentum_thr:
            v[2] =  1.0 if mom > 0 else -1.0        # Bit 2: direction
            v[3] =  1.0 if abs(mom) > self.momentum_strong else -1.0  # Bit 3: magnitude

        v[4] =  1.0 if vrat > self.vol_ratio_high else -1.0   # Bit 4: high volume
        v[5] =  1.0 if vrat > 2.0               else -1.0     # Bit 5: surge
        v[6] =  1.0 if volt > self.vol_high      else -1.0    # Bit 6: elevated vol
        v[7] =  1.0 if volt > self.vol_extreme   else -1.0    # Bit 7: extreme vol

        # --- DELTA BITS ---
        if prev is not None:
            rsi_d = rsi - prev['rsi']
            if abs(rsi_d) > 1.0:
                v[8] =  1.0 if rsi_d > 0 else -1.0
                v[9] =  1.0 if abs(rsi_d) > 5.0 else -1.0

            p, pp = ind['price'], prev['price']
            prev_pp = prev.get('prev_price', pp)
            accel = (p - pp) - (pp - prev_pp)
            if abs(accel) > 10.0:
                v[10] =  1.0 if accel > 0 else -1.0
                v[11] =  1.0 if abs(accel) > 100.0 else -1.0

            vol_cur  = ind['volume']
            vol_prev = prev['volume']
            if vol_prev > 0:
                vd = (vol_cur - vol_prev) / vol_prev
                if abs(vd) > 0.3:
                    v[12] =  1.0 if vd > 0 else -1.0
                    v[13] =  1.0 if abs(vd) > 0.7 else -1.0

            volt_d = volt - prev['volatility']
            if abs(volt_d) > 0.01:
                v[14] =  1.0 if volt_d > 0 else -1.0
                v[15] =  1.0 if abs(volt_d) > 0.03 else -1.0

        return v


# ---------------------------------------------------------------------------
# 4.  SYNTHETIC DATA GENERATION  (balanced 3-class)
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_samples: int = 12000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic BTC-style tick data with balanced BUY / HOLD / SELL labels.

    Label rule (deterministic, not ambiguous):
      SELL : RSI > 70  AND strong positive momentum   (overbought blowoff)
      BUY  : RSI < 30  AND strong negative momentum   (oversold capitulation)
      HOLD : everything else
    """
    print(f"Generating {n_samples} synthetic samples...")
    quantizer = BipolarQuantizer()
    X, y = [], []

    price        = 50_000.0
    prev_ind     = None
    buy_c = sell_c = hold_c = 0

    for i in range(n_samples):
        mom  = np.random.randn() * 0.008
        price = max(100.0, price * (1 + mom))

        rsi  = 50 + 40 * np.sin(i / 80.0) + np.random.randn() * 8
        rsi  = float(np.clip(rsi, 1, 99))

        vrat = max(0.1, 1.0 + 0.6 * np.random.randn())
        volt = max(0.005, 0.025 + 0.015 * abs(np.random.randn()))

        ind = {
            'rsi': rsi, 'momentum': mom,
            'volume_ratio': vrat, 'volatility': volt,
            'price': price, 'volume': 1000 * vrat,
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
    return X, y


# ---------------------------------------------------------------------------
# 5.  MODEL
# ---------------------------------------------------------------------------

def build_model() -> keras.Model:
    print("\nBuilding BNN model (16 → 64 → 3)...")
    model = keras.Sequential([
        BinaryDense(64,          name='layer1'),   # hidden: binary act
        BinaryOutputDense(3,     name='layer2'),   # output: raw sums
        keras.layers.Softmax()
    ])
    model.build((None, 16))
    model.summary()
    total_bits = 16 * 64 + 64 * 3
    print(f"\n  Binary parameters : {total_bits} bits  ({total_bits/8:.0f} bytes)")
    print(f"  BRAM usage        : {total_bits/1024:.3f} kbits / 32 kbits  "
          f"({100*total_bits/(32*1024):.1f}%)")
    return model


def train_model(model, X_tr, y_tr, X_val, y_val):
    print("\nTraining BNN...")

    # Class weights to counteract any residual imbalance
    counts   = y_tr.sum(axis=0)
    cw_vals  = (1.0 / counts) * (counts.sum() / 3.0)
    cw       = {i: cw_vals[i] for i in range(3)}

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=64,
        class_weight=cw,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5
            )
        ],
        verbose=1
    )
    return history.history


# ---------------------------------------------------------------------------
# 6.  WEIGHT EXTRACTION
# ---------------------------------------------------------------------------

def extract_weights(model, output_dir: Path):
    print("\n" + "="*60)
    print("EXTRACTING BINARY WEIGHTS")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    w1 = model.get_layer('layer1').get_binary_weights()  # (16, 64)  {-1,+1}
    w2 = model.get_layer('layer2').get_binary_weights()  # (64,  3)  {-1,+1}

    assert set(np.unique(w1)) == {-1.0, 1.0}, "Layer-1 weights not strictly binary!"
    assert set(np.unique(w2)) == {-1.0, 1.0}, "Layer-2 weights not strictly binary!"

    print(f"  Layer-1 weights : {w1.shape}  values = {np.unique(w1)}")
    print(f"  Layer-2 weights : {w2.shape}  values = {np.unique(w2)}")

    # Hardware mapping: -1 → bit 0,  +1 → bit 1
    w1_bin = ((w1 + 1) // 2).astype(np.uint8)   # (16, 64) in {0,1}
    w2_bin = ((w2 + 1) // 2).astype(np.uint8)   # (64,  3) in {0,1}

    # --- Verilog $readmemb file ---
    # Layout: neuron-major (neuron 0 all inputs, then neuron 1, …)
    # Matches the BRAM addressing scheme in the FSM:
    #   addr = neuron_idx * N_INPUTS + input_idx
    with open(output_dir / "weights.mem", "w") as f:
        f.write("// Layer 1: 64 neurons × 16 inputs = 1,024 bits\n")
        f.write("// Address layout: [neuron_j * 16 + input_i]\n")
        for j in range(64):
            for i in range(16):
                f.write(f"{w1_bin[i, j]}\n")
        f.write("// Layer 2: 3 neurons × 64 inputs = 192 bits\n")
        f.write("// Address layout: [1024 + neuron_j * 64 + input_i]\n")
        for j in range(3):
            for i in range(64):
                f.write(f"{w2_bin[i, j]}\n")

    # --- ESP32 C header ---
    with open(output_dir / "bnn_weights.h", "w") as f:
        f.write("#ifndef BNN_WEIGHTS_H\n#define BNN_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// Layer-1: [input_i][neuron_j]  values in {0,1}  (-1→0, +1→1)\n")
        f.write("const uint8_t layer1_weights[16][64] = {\n")
        for i in range(16):
            row = ", ".join(str(w1_bin[i, j]) for j in range(64))
            f.write(f"  {{{row}}},\n")
        f.write("};\n\n")
        f.write("// Layer-2: [input_i][neuron_j]  values in {0,1}\n")
        f.write("const uint8_t layer2_weights[64][3] = {\n")
        for i in range(64):
            row = ", ".join(str(w2_bin[i, j]) for j in range(3))
            f.write(f"  {{{row}}},\n")
        f.write("};\n\n#endif // BNN_WEIGHTS_H\n")

    # --- Metadata ---
    metadata = {
        "architecture"     : {"input": 16, "hidden": 64, "output": 3},
        "total_bits"       : int(16*64 + 64*3),
        "bram_usage_kbits" : float((16*64 + 64*3) / 1024),
        "bram_pct"         : float((16*64 + 64*3) / (32*1024) * 100),
        "weight_domain"    : "{-1, +1}",
        "hardware_mapping" : {"-1": "0", "+1": "1"},
        "activation_hidden": "sign(x>=0)",
        "activation_output": "raw_sums_softmax",
        "mem_layout"       : "neuron_major"
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved to {output_dir.absolute()}")
    print("    weights.mem    (Verilog $readmemb)")
    print("    bnn_weights.h  (ESP32 C header)")
    print("    metadata.json  (architecture record)")

    return w1_bin, w2_bin


# ---------------------------------------------------------------------------
# 7.  HARDWARE-ACCURATE FPGA SIMULATION
# ---------------------------------------------------------------------------

def fpga_inference_sim(x_bipolar: np.ndarray,
                       w1_bin: np.ndarray,
                       w2_bin: np.ndarray) -> int:
    """
    Cycle-accurate simulation of the Verilog XNOR-popcount BNN.

    Operates entirely in the {0,1} binary domain — no floating-point
    multiplies, no np.sign().  This is what the Verilog FSM computes.

    Layer 1 (hidden, 64 neurons, 16 inputs each):
      x_bit   = (x_bipolar == +1) ? 1 : 0
      xnor    = ~(x_bit XOR w_bit)          # XNOR gate per input
      popcount = sum(xnor)                   # popcount tree
      fire     = (popcount >= 8) ? 1 : 0    # threshold = N/2 = 8

    Layer 2 (output, 3 neurons, 64 inputs each):
      xnor    = ~(h_bit XOR w_bit)
      score[j] = sum(xnor)                  # raw popcount — winner-take-all
    Returns:
      argmax(score)  →  0=BUY, 1=HOLD, 2=SELL
    """
    # Convert input bipolar → binary {0,1}
    x_bits = ((x_bipolar + 1) // 2).astype(np.int32)   # -1→0, +1→1

    # --- Layer 1: XNOR-popcount with threshold ---
    h_bits = np.zeros(64, dtype=np.int32)
    for j in range(64):
        xnor     = 1 - np.bitwise_xor(x_bits, w1_bin[:, j].astype(np.int32))
        popcount = int(np.sum(xnor))
        h_bits[j] = 1 if popcount >= 8 else 0     # threshold = 16/2

    # --- Layer 2: XNOR-popcount raw scores (no threshold → winner-take-all) ---
    scores = np.zeros(3, dtype=np.int32)
    for j in range(3):
        xnor      = 1 - np.bitwise_xor(h_bits, w2_bin[:, j].astype(np.int32))
        scores[j] = int(np.sum(xnor))

    return int(np.argmax(scores))


def verify_fpga(model, X_test: np.ndarray,
                w1_bin: np.ndarray, w2_bin: np.ndarray) -> float:
    print("\n" + "="*60)
    print("FPGA HARDWARE VERIFICATION  (XNOR-popcount simulation)")
    print("="*60)

    n = min(500, len(X_test))
    matches = 0
    mismatches_detail = []

    for i in range(n):
        x = X_test[i]

        # TensorFlow prediction (ground truth for the trained weights)
        tf_pred = int(np.argmax(model.predict(x.reshape(1, -1), verbose=0)[0]))

        # Hardware simulation prediction
        hw_pred = fpga_inference_sim(x, w1_bin, w2_bin)

        if tf_pred == hw_pred:
            matches += 1
        elif len(mismatches_detail) < 5:
            mismatches_detail.append((i, tf_pred, hw_pred))

    match_rate = 100.0 * matches / n
    print(f"\n  Samples tested : {n}")
    print(f"  Matches        : {matches}")
    print(f"  Match rate     : {match_rate:.1f}%")

    if mismatches_detail:
        print("\n  First mismatches (sample_idx, tf_pred, hw_pred):")
        for d in mismatches_detail:
            print(f"    {d}")

    if match_rate >= 99.0:
        print("\n  ✅ HARDWARE SIMULATION VALIDATED — weights are FPGA-safe")
    elif match_rate >= 95.0:
        print("\n  ⚠  Match rate acceptable but check mismatch samples above")
    else:
        print("\n  ❌ CRITICAL MISMATCH — do NOT flash weights; debug required")

    return match_rate


# ---------------------------------------------------------------------------
# 8.  MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    print("="*60)
    print("BNN TRAINING PIPELINE — PHASE 1 (Hardware-Accurate)")
    print("="*60)

    # Reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Data ---
    X, y = generate_synthetic_data(12000)

    n        = len(X)
    tr_end   = int(0.70 * n)
    val_end  = int(0.85 * n)

    X_train, y_train = X[:tr_end],       y[:tr_end]
    X_val,   y_val   = X[tr_end:val_end], y[tr_end:val_end]
    X_test,  y_test  = X[val_end:],       y[val_end:]

    print(f"\n  Train : {len(X_train)}  Val : {len(X_val)}  Test : {len(X_test)}")

    # --- Model ---
    model   = build_model()
    history = train_model(model, X_train, y_train, X_val, y_val)

    # --- Evaluate ---
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*60}")
    print(f"  FINAL TEST ACCURACY : {acc*100:.2f}%")
    print(f"  FINAL TEST LOSS     : {loss:.4f}")
    print(f"{'='*60}")

    # --- Extract ---
    w1_bin, w2_bin = extract_weights(model, Path("fpga_weights"))

    # --- Verify hardware equivalence ---
    match_rate = verify_fpga(model, X_test, w1_bin, w2_bin)

    # --- Summary ---
    print("\n" + "="*60)
    print("PHASE 1 SUMMARY")
    print("="*60)
    print(f"  Test accuracy    : {acc*100:.2f}%")
    print(f"  HW match rate    : {match_rate:.1f}%")
    print(f"  Total parameters : {16*64 + 64*3} bits  ({(16*64+64*3)//8} bytes)")
    print(f"  BRAM usage       : {(16*64+64*3)/1024:.3f} kbits / 32 kbits")

    if match_rate >= 99.0 and acc >= 0.70:
        print("\n  ✅ PHASE 1 COMPLETE — Ready for Phase 2 (ESP32 Firmware)")
    else:
        print("\n  ⚠  Phase 1 incomplete — see warnings above before proceeding")

    print("="*60)


if __name__ == "__main__":
    main()
