import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# --- Ternary Quantization with STE ---
class TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=0.05):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        out[input < -threshold] = -1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (input.abs() <= 1.0).float(), None

def ternary_quantize(x, threshold=0.05):
    return TernarySTE.apply(x, threshold)

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, threshold=0.05, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-0.5, 0.5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, input):
        w_t = ternary_quantize(self.weight, self.threshold)
        return torch.nn.functional.linear(input, w_t, bias=self.bias)
        
    def get_ternary_weights(self):
        w_t = ternary_quantize(self.weight, self.threshold)
        return w_t.detach().cpu().numpy()
        
    def get_bias(self):
        if self.bias is not None:
            return self.bias.detach().cpu().numpy()
        return np.zeros(self.out_features)

class BinaryActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.ones_like(input)
        out[input < 0] = -1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def binary_activation(x):
    return BinaryActivationSTE.apply(x)

# --- Synthetic UDP Jitter (Temporal Dropout) ---
class TemporalDropout(nn.Module):
    def __init__(self, drop_prob=0.05):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training:
            # x shape: (batch_size, 40)
            # 4 ticks x 10 features.
            # Tick 0 is [0:10], Tick -1 is [10:20], Tick -2 is [20:30], Tick -3 is [30:40]
            mask = torch.ones_like(x)
            
            # Randomly drop t-1 (idx 1) or t-2 (idx 2)
            drop_t1 = (torch.rand(x.size(0), device=x.device) < self.drop_prob)
            drop_t2 = (torch.rand(x.size(0), device=x.device) < self.drop_prob)
            
            mask[drop_t1, 10:20] = 0.0
            mask[drop_t2, 20:30] = 0.0
            
            return x * mask
        return x

class PredictiveMicrostructureBNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 40 Inputs (4 ticks * 10 features) -> 32 Hidden -> 3 Output (BUY, HOLD, SELL)
        self.dropout = TemporalDropout(drop_prob=0.05)
        self.fc1 = TernaryLinear(40, 32, threshold=0.1, bias=True)
        self.fc2 = TernaryLinear(32, 3, threshold=0.1, bias=True)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = binary_activation(x)
        x = self.fc2(x)
        return x

def generate_synthetic_data(regime='momentum', n_samples=12000, k=5):
    print(f"Generating {n_samples} synthetic samples for regime: {regime}...")
    prices = np.zeros(n_samples)
    prices[0] = 50000.0
    
    vol = 4.0 if regime == 'momentum' else 1.0
    for i in range(1, n_samples):
        prices[i] = prices[i-1] + (np.random.randn() * vol)
        if regime == 'ranging':
            # Mean reversion pull
            prices[i] += (50000.0 - prices[i]) * 0.05
        
    X = []
    y = []
    
    for i in range(n_samples - k):
        M_t = prices[i]
        M_t_k = prices[i + k]
        ret = M_t_k - M_t
        
        if regime == 'momentum':
            if ret > 2.0: label = 1.0 # Buy
            elif ret < -2.0: label = 0.0 # Sell
            else: label = 2.0 # Hold
        else:
            if ret > 0.5: label = 1.0 # Buy
            elif ret < -0.5: label = 0.0 # Sell
            else: label = 2.0 # Hold
            
        temporal_spike = np.full(40, -1.0, dtype=np.float32)
        
        # We fill 4 ticks backward. For synthesis, we'll just populate them logically.
        for t in range(4):
            spike = np.full(10, -1.0, dtype=np.float32)
            # Probabilistic feature assignment to make the task harder
            # This prevents the BNN from collapsing into a 2-input OR gate
            if label == 1.0: 
                if np.random.rand() > 0.3: spike[1] = 1.0 # OFI > 0
                if np.random.rand() > 0.7: spike[0] = 1.0
                if np.random.rand() > 0.4: spike[4] = 1.0 # Mid > VWAP
                if np.random.rand() > 0.3: spike[6] = 1.0 # Lee-Ready BUYER
            elif label == 0.0:
                if np.random.rand() > 0.3: spike[3] = 1.0 # OFI < 0
                if np.random.rand() > 0.7: spike[2] = 1.0
                if np.random.rand() > 0.4: spike[5] = 1.0 # Mid < VWAP
                if np.random.rand() > 0.3: spike[7] = 1.0 # Lee-Ready SELLER
                
            # Velocity Bits [8, 9]
            # 00 = slow, 01 = norm, 10 = fast, 11 = very fast
            vel = np.random.randint(0, 4)
            if vel & 1: spike[8] = 1.0
            if vel & 2: spike[9] = 1.0
            
            # Noise
            for b in range(10):
                if np.random.rand() < 0.2:
                    spike[b] = -spike[b]
                    
            temporal_spike[(t*10):(t*10+10)] = spike
        
        X.append(temporal_spike)
        y.append(label)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.longlong)
    return X, y

def l1_regularization(model, lambda_l1):
    l1_loss = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

def train_regime(model, X, y, device, epochs=150, lambda_l1=0.05, lr=0.01):
    X = torch.tensor(X).to(device)
    y = torch.tensor(y).to(device)
    
    # Filter parameters to only train ones that require grad
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X)
        ce_loss = criterion(logits, y)
        reg_loss = l1_regularization(model, lambda_l1)
        loss = ce_loss + reg_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = torch.argmax(model(X), dim=1)
                acc = (preds == y).float().mean().item()
                w1 = model.fc1.get_ternary_weights()
                sparsity = np.sum(w1 == 0) / w1.size
                print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc*100:.1f}% | Sparsity: {sparsity*100:.1f}%")
                
    return model

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training anchored to device: {device}")
    
    # --- PHASE A: Train Momentum Expert (Find Sparsity Mask + Bias A) ---
    print("\n--- Training Model A (Momentum Expert) ---")
    X_A, y_A = generate_synthetic_data(regime='momentum')
    model_A = PredictiveMicrostructureBNN().to(device)
    model_A = train_regime(model_A, X_A, y_A, device, epochs=150, lambda_l1=0.05, lr=0.01)
    
    w1 = model_A.fc1.get_ternary_weights()
    w2 = model_A.fc2.get_ternary_weights()
    b1_A = model_A.fc1.get_bias()
    b2_A = model_A.fc2.get_bias()
    
    # --- PHASE B: Train Ranging Expert (Reuse Sparsity Mask, Learn Bias B) ---
    print("\n--- Training Model B (Ranging Expert) ---")
    X_B, y_B = generate_synthetic_data(regime='ranging')
    model_B = PredictiveMicrostructureBNN().to(device)
    
    # Copy exact weights from A to B
    model_B.fc1.weight.data = model_A.fc1.weight.data.clone()
    model_B.fc2.weight.data = model_A.fc2.weight.data.clone()
    
    # FREEZE the weights so the wire routing remains identical
    model_B.fc1.weight.requires_grad = False
    model_B.fc2.weight.requires_grad = False
    
    # Train B (L1 is 0 since weights are frozen)
    model_B = train_regime(model_B, X_B, y_B, device, epochs=150, lambda_l1=0.0, lr=0.02)
    
    b1_B = model_B.fc1.get_bias()
    b2_B = model_B.fc2.get_bias()
    
    # Save the MoE weights to npz
    output_dir = Path("fpga_weights")
    output_dir.mkdir(exist_ok=True)
    np.savez(output_dir / "ternary_weights.npz", 
             w1=w1, w2=w2, 
             b1_A=b1_A, b2_A=b2_A,
             b1_B=b1_B, b2_B=b2_B)
    print(f"\nSaved MoE weights to {output_dir / 'ternary_weights.npz'}")

if __name__ == "__main__":
    main()
