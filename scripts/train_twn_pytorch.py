import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from pathlib import Path

# --- Ternary Quantization with STE ---
class TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=0.05):
        # Quantize to {-1, 0, 1}
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        out[input < -threshold] = -1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator
        return grad_output, None

def ternary_quantize(x, threshold=0.05):
    return TernarySTE.apply(x, threshold)

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, threshold=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-0.5, 0.5))
        
    def forward(self, input):
        # Binarize inputs if they aren't already (our inputs are -1 or 1, but intermediate activations need it)
        # Actually, for the input layer, inputs are strictly {-1, 1}.
        # For hidden activations, they will be binary {-1, 1} too.
        w_t = ternary_quantize(self.weight, self.threshold)
        return torch.nn.functional.linear(input, w_t, bias=None)
        
    def get_ternary_weights(self):
        w_t = ternary_quantize(self.weight, self.threshold)
        return w_t.detach().cpu().numpy()

class BinaryActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Output +1 if >= 0 else -1
        out = torch.ones_like(input)
        out[input < 0] = -1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def binary_activation(x):
    return BinaryActivationSTE.apply(x)

class PredictiveMicrostructureBNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 8 Inputs -> 32 Hidden (Ternary) -> 1 Output (Ternary weights, but binary activation out? Or raw sum for BCE?)
        self.fc1 = TernaryLinear(8, 32, threshold=0.1)
        self.fc2 = TernaryLinear(32, 1, threshold=0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = binary_activation(x)
        x = self.fc2(x)
        # Return raw sum (logits). We use BCEWithLogitsLoss.
        return x

def generate_synthetic_data(n_samples=12000, k=5):
    print(f"Generating {n_samples} synthetic samples...")
    # Simulate a midprice random walk with mean reversion
    prices = np.zeros(n_samples)
    prices[0] = 50000.0
    for i in range(1, n_samples):
        prices[i] = prices[i-1] + np.random.randn() * 2.0
        
    X = []
    y = []
    buy_c = 0
    sell_c = 0
    
    for i in range(n_samples - k):
        # Forward return
        M_t = prices[i]
        M_t_k = prices[i + k]
        ret = M_t_k - M_t
        
        # Target Label
        if ret > 1.0:
            label = 1.0
            buy_c += 1
        elif ret < -1.0:
            label = 0.0
            sell_c += 1
        else:
            continue # Skip flat for binary classification
            
        # Features (simulate OFI, Microprice, Lee-Ready correlations)
        # Bit 0: OFI > Strong Positive
        # Bit 1: OFI > 0
        # Bit 2: OFI < Strong Negative
        # Bit 3: OFI < 0
        # Bit 4: Midpoint > VWAP (Upward pressure)
        # Bit 5: Midpoint < VWAP (Downward pressure)
        # Bit 6: Lee-Ready == BUYER
        # Bit 7: Lee-Ready == SELLER
        
        spike = np.full(8, -1.0, dtype=np.float32)
        if label == 1.0: # Buy
            spike[1] = 1.0 # OFI > 0
            if np.random.rand() > 0.5: spike[0] = 1.0
            spike[4] = 1.0 # Mid > VWAP
            spike[6] = 1.0 # Lee-Ready BUYER
        else:
            spike[3] = 1.0 # OFI < 0
            if np.random.rand() > 0.5: spike[2] = 1.0
            spike[5] = 1.0 # Mid < VWAP
            spike[7] = 1.0 # Lee-Ready SELLER
            
        # Add noise
        for b in range(8):
            if np.random.rand() < 0.2:
                spike[b] = -spike[b]
                
        X.append(spike)
        y.append([label])
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"Generated {len(X)} valid samples (BUY: {buy_c}, SELL: {sell_c})")
    return X, y

def l1_regularization(model, lambda_l1):
    l1_loss = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    X, y = generate_synthetic_data()
    X = torch.tensor(X).to(device)
    y = torch.tensor(y).to(device)
    
    model = PredictiveMicrostructureBNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    lambda_l1 = 0.05  # Strong L1 to push weights to 0
    
    epochs = 200
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X)
        bce_loss = criterion(logits, y)
        reg_loss = l1_regularization(model, lambda_l1)
        loss = bce_loss + reg_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            # Eval acc
            preds = (logits > 0.0).float()
            acc = (preds == y).float().mean().item()
            
            w1 = model.fc1.get_ternary_weights()
            w2 = model.fc2.get_ternary_weights()
            sparsity = (np.sum(w1 == 0) + np.sum(w2 == 0)) / (w1.size + w2.size)
            
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc*100:.1f}% | Sparsity: {sparsity*100:.1f}%")

    w1 = model.fc1.get_ternary_weights() # shape: (32, 8)
    w2 = model.fc2.get_ternary_weights() # shape: (1, 32)
    
    # Save the trained weights to npz for the generator script
    output_dir = Path("fpga_weights")
    output_dir.mkdir(exist_ok=True)
    np.savez(output_dir / "ternary_weights.npz", w1=w1, w2=w2)
    print(f"Saved weights to {output_dir / 'ternary_weights.npz'}")
    
if __name__ == "__main__":
    train()
