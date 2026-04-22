"""
ANN Assignment 2
Models: ISNN-1 and ISNN-2
Reference: arXiv:2503.00268

Two implementations:
  1. PyTorch  (using autograd)
  2. NumPy    (manual forward + manual backpropagation)

Toy problems:
  Problem 1: f = exp(-0.5x) + log(1+exp(0.4y)) + tanh(t) + sin(z) - 0.4
  Problem 2: g = exp(-0.3x) * (0.15y)^2 * tanh(0.3t) * (0.2*sin(0.5z+2)+0.5)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (saves to file without needing a display)
import matplotlib.pyplot as plt
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────
# Activation functions for NumPy manual implementation
# ─────────────────────────────────────────────────────────────

def softplus(x):
    # sigma_mc from the paper: log(1 + exp(x))
    # Numerically stable: for large x, softplus(x) ≈ x
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))

def softplus_grad(x):
    # derivative of softplus(x) is sigmoid(x)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid(x):
    # sigma_m = sigma_a from the paper
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)


# ─────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────

def lhs_sample(n_samples, n_dims, low, high, seed=0):
    """Latin Hypercube Sampling."""
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    pts = sampler.random(n=n_samples)
    return qmc.scale(pts, low, high)

def toy_problem_1(x, y, t, z):
    # convex in x, convex+monotone in y, monotone in t, arbitrary in z
    return np.exp(-0.5 * x) + np.log(1.0 + np.exp(0.4 * y)) + np.tanh(t) + np.sin(z) - 0.4

def toy_problem_2(x, y, z, t):
    fx = np.exp(-0.3 * x)
    fy = (0.15 * y) ** 2
    ft = np.tanh(0.3 * t)
    fz = 0.2 * np.sin(0.5 * z + 2) + 0.5
    return fx * fy * fz * ft

def generate_dataset(problem=1):
    # 500 training samples via LHS in [0, 4]^4
    X_train = lhs_sample(500, 4, 0.0, 4.0, seed=0)

    # 5000 test samples, wider range
    test_high = 6.0 if problem == 1 else 10.0
    rng = np.random.default_rng(1)
    X_test = rng.uniform(0.0, test_high, (5000, 4))

    x_tr, y_tr, t_tr, z_tr = X_train[:, 0], X_train[:, 1], X_train[:, 2], X_train[:, 3]
    x_te, y_te, t_te, z_te = X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3]

    if problem == 1:
        f_train = toy_problem_1(x_tr, y_tr, t_tr, z_tr)
        f_test  = toy_problem_1(x_te, y_te, t_te, z_te)
    else:
        f_train = toy_problem_2(x_tr, y_tr, z_tr, t_tr)
        f_test  = toy_problem_2(x_te, y_te, z_te, t_te)

    return (X_train.astype(np.float32),
            f_train.astype(np.float32),
            X_test.astype(np.float32),
            f_test.astype(np.float32))


# ─────────────────────────────────────────────────────────────
# PYTORCH IMPLEMENTATION
# ─────────────────────────────────────────────────────────────
#
# Inputs are split as:
#   x0 = X[:, 0]  — convex constraint
#   y0 = X[:, 1]  — convex + monotone
#   t0 = X[:, 2]  — monotone only
#   z0 = X[:, 3]  — arbitrary (no constraint)
#
# Weight constraints are enforced by passing raw parameters
# through F.softplus before multiplying, ensuring W_actual >= 0.
# ─────────────────────────────────────────────────────────────

class ISNN1_PyTorch(nn.Module):
    """
    ISNN-1: separate branches for each input group.
    y, z, t branches finish first, then feed into the x branch at layer 0.
    Architecture: 2 hidden layers per branch, 10 neurons per layer.
    """

    def __init__(self, n=10):
        super().__init__()

        # ── y branch (convex + monotone): non-negative weights, softplus activation
        self.yw0 = nn.Parameter(torch.randn(1, n) * 0.1)   # 1→n
        self.yb0 = nn.Parameter(torch.zeros(n))
        self.yw1 = nn.Parameter(torch.randn(n, n) * 0.1)   # n→n  (non-neg)
        self.yb1 = nn.Parameter(torch.zeros(n))

        # ── z branch (arbitrary): unconstrained weights, sigmoid activation
        self.zw0 = nn.Parameter(torch.randn(1, n) * 0.1)
        self.zb0 = nn.Parameter(torch.zeros(n))
        self.zw1 = nn.Parameter(torch.randn(n, n) * 0.1)
        self.zb1 = nn.Parameter(torch.zeros(n))

        # ── t branch (monotone): non-negative weights, sigmoid activation
        self.tw0 = nn.Parameter(torch.randn(1, n) * 0.1)
        self.tb0 = nn.Parameter(torch.zeros(n))
        self.tw1 = nn.Parameter(torch.randn(n, n) * 0.1)   # non-neg
        self.tb1 = nn.Parameter(torch.zeros(n))

        # ── x branch (convex)
        # Layer 0: receives x0, y_out, z_out, t_out.
        #   W0_xx at h=0 has no non-neg constraint (paper, eq 5: h=1,...,Hx-1)
        self.xw0_xx = nn.Parameter(torch.randn(1, n) * 0.1)
        self.xw0_xy = nn.Parameter(torch.randn(n, n) * 0.1)  # non-neg
        self.xw0_xz = nn.Parameter(torch.randn(n, n) * 0.1)  # free
        self.xw0_xt = nn.Parameter(torch.randn(n, n) * 0.1)  # non-neg
        self.xb0    = nn.Parameter(torch.zeros(n))

        # Layer 1: only x, non-negative W_xx (h=1 >= 1)
        self.xw1 = nn.Parameter(torch.randn(n, n) * 0.1)
        self.xb1 = nn.Parameter(torch.zeros(n))

        # Output layer
        self.w_out = nn.Parameter(torch.randn(n, 1) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        x0 = X[:, 0:1]
        y0 = X[:, 1:2]
        t0 = X[:, 2:3]
        z0 = X[:, 3:4]

        # y branch — softplus activation, non-neg weights
        y1 = F.softplus(y0 @ F.softplus(self.yw0) + self.yb0)
        y2 = F.softplus(y1 @ F.softplus(self.yw1) + self.yb1)

        # z branch — sigmoid activation, free weights
        z1 = torch.sigmoid(z0 @ self.zw0 + self.zb0)
        z2 = torch.sigmoid(z1 @ self.zw1 + self.zb1)

        # t branch — sigmoid activation, non-neg weights
        t1 = torch.sigmoid(t0 @ F.softplus(self.tw0) + self.tb0)
        t2 = torch.sigmoid(t1 @ F.softplus(self.tw1) + self.tb1)

        # x branch layer 0 (eq. 4): mixes x0 with final branch outputs
        x1 = F.softplus(
            x0 @ self.xw0_xx +
            y2 @ F.softplus(self.xw0_xy) +
            z2 @ self.xw0_xz +
            t2 @ F.softplus(self.xw0_xt) +
            self.xb0
        )

        # x branch layer 1 (eq. 5): only x, non-neg weights
        x2 = F.softplus(x1 @ F.softplus(self.xw1) + self.xb1)

        return x2 @ self.w_out + self.b_out


class ISNN2_PyTorch(nn.Module):
    """
    ISNN-2: skip/pass-through connections.
    Each x layer sees the current x, the original x0, and the current
    state of all branch hidden layers.
    Architecture: 1 hidden layer per branch, 15 neurons.
    """

    def __init__(self, n=15):
        super().__init__()

        # ── y branch (1 hidden layer)
        self.yw0 = nn.Parameter(torch.randn(1, n) * 0.1)   # non-neg
        self.yb0 = nn.Parameter(torch.zeros(n))

        # ── z branch (1 hidden layer)
        self.zw0 = nn.Parameter(torch.randn(1, n) * 0.1)   # free
        self.zb0 = nn.Parameter(torch.zeros(n))

        # ── t branch (1 hidden layer)
        self.tw0 = nn.Parameter(torch.randn(1, n) * 0.1)   # non-neg
        self.tb0 = nn.Parameter(torch.zeros(n))

        # ── x layer 0 (eq. 9): takes raw x0, y0, z0, t0 directly
        self.xw0_xx = nn.Parameter(torch.randn(1, n) * 0.1)   # free at h=0
        self.xw0_xy = nn.Parameter(torch.randn(1, n) * 0.1)   # non-neg (from raw y0)
        self.xw0_xz = nn.Parameter(torch.randn(1, n) * 0.1)   # free
        self.xw0_xt = nn.Parameter(torch.randn(1, n) * 0.1)   # non-neg (from raw t0)
        self.xb0    = nn.Parameter(torch.zeros(n))

        # ── x layer 1 (eq. 10): x1, x0 skip, y1, z1, t1
        self.xw1_xx  = nn.Parameter(torch.randn(n, n) * 0.1)  # non-neg (h=1 >= 1)
        self.xw1_xx0 = nn.Parameter(torch.randn(1, n) * 0.1)  # skip from original x0
        self.xw1_xy  = nn.Parameter(torch.randn(n, n) * 0.1)  # non-neg
        self.xw1_xz  = nn.Parameter(torch.randn(n, n) * 0.1)  # free
        self.xw1_xt  = nn.Parameter(torch.randn(n, n) * 0.1)  # non-neg
        self.xb1     = nn.Parameter(torch.zeros(n))

        # Output
        self.w_out = nn.Parameter(torch.randn(n, 1) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        x0 = X[:, 0:1]
        y0 = X[:, 1:2]
        t0 = X[:, 2:3]
        z0 = X[:, 3:4]

        # Branch hidden states (computed from raw inputs)
        y1 = F.softplus(y0 @ F.softplus(self.yw0) + self.yb0)
        z1 = torch.sigmoid(z0 @ self.zw0 + self.zb0)
        t1 = torch.sigmoid(t0 @ F.softplus(self.tw0) + self.tb0)

        # x layer 0 — uses raw inputs (eq. 9)
        x1 = F.softplus(
            x0 @ self.xw0_xx +
            y0 @ F.softplus(self.xw0_xy) +
            z0 @ self.xw0_xz +
            t0 @ F.softplus(self.xw0_xt) +
            self.xb0
        )

        # x layer 1 — skip connections to x0 and branch states (eq. 10)
        x2 = F.softplus(
            x1 @ F.softplus(self.xw1_xx) +
            x0 @ self.xw1_xx0 +
            y1 @ F.softplus(self.xw1_xy) +
            z1 @ self.xw1_xz +
            t1 @ F.softplus(self.xw1_xt) +
            self.xb1
        )

        return x2 @ self.w_out + self.b_out


def train_pytorch(model, X_train, y_train, X_test, y_test,
                  epochs=5000, lr=1e-3):
    """Standard training loop using Adam, returns loss histories."""
    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train).unsqueeze(1)
    Xte = torch.from_numpy(X_test)
    yte = torch.from_numpy(y_test).unsqueeze(1)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(Xtr)
        loss = F.mse_loss(pred, ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            tloss = F.mse_loss(model(Xte), yte).item()

        train_losses.append(loss.item())
        test_losses.append(tloss)

        if (epoch + 1) % 1000 == 0:
            print(f"  epoch {epoch+1:5d} | train MSE: {loss.item():.6f} | test MSE: {tloss:.6f}")

    return train_losses, test_losses


# ─────────────────────────────────────────────────────────────
# NUMPY MANUAL IMPLEMENTATION
# ─────────────────────────────────────────────────────────────
#
# Backpropagation follows the chain rule exactly as derived from
# the MSE loss through each layer. Constrained parameters (raw)
# are passed through softplus before use, so an extra chain-rule
# term softplus_grad(raw_param) appears in their gradient.
#
# Adam is implemented manually with the standard update rule.
# ─────────────────────────────────────────────────────────────

class AdamState:
    """Tracks first/second moment buffers for every parameter in a dict."""

    def __init__(self, param_dict, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m = {k: np.zeros_like(v) for k, v in param_dict.items()}
        self.v = {k: np.zeros_like(v) for k, v in param_dict.items()}

    def step(self, param_dict, grad_dict):
        self.t += 1
        t = self.t
        for k in param_dict:
            g = grad_dict[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2
            m_hat = self.m[k] / (1 - self.beta1 ** t)
            v_hat = self.v[k] / (1 - self.beta2 ** t)
            param_dict[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ISNN1_NumPy:
    """
    ISNN-1 implemented with NumPy only.
    All operations are explicit matrix multiplications.
    Backpropagation is written out manually layer by layer.
    """

    def __init__(self, n=10, seed=42):
        rng = np.random.RandomState(seed)
        s = 0.1

        self.params = {
            # y branch (raw params — softplus applied during forward)
            'yw0': rng.randn(1, n) * s,  # 1 → n
            'yb0': np.zeros(n),
            'yw1': rng.randn(n, n) * s,  # n → n
            'yb1': np.zeros(n),

            # z branch (unconstrained)
            'zw0': rng.randn(1, n) * s,
            'zb0': np.zeros(n),
            'zw1': rng.randn(n, n) * s,
            'zb1': np.zeros(n),

            # t branch (raw params — softplus applied during forward)
            'tw0': rng.randn(1, n) * s,
            'tb0': np.zeros(n),
            'tw1': rng.randn(n, n) * s,
            'tb1': np.zeros(n),

            # x branch
            'xw0_xx': rng.randn(1, n) * s,   # free (h=0)
            'xw0_xy': rng.randn(n, n) * s,   # non-neg (raw)
            'xw0_xz': rng.randn(n, n) * s,   # free
            'xw0_xt': rng.randn(n, n) * s,   # non-neg (raw)
            'xb0':    np.zeros(n),
            'xw1':    rng.randn(n, n) * s,   # non-neg (raw), h=1
            'xb1':    np.zeros(n),

            # output
            'w_out': rng.randn(n, 1) * s,
            'b_out': np.zeros(1),
        }
        self.adam = AdamState(self.params)
        self.cache = {}

    def forward(self, X):
        p = self.params
        x0 = X[:, 0:1]
        y0 = X[:, 1:2]
        t0 = X[:, 2:3]
        z0 = X[:, 3:4]

        # Constrained (actual) weight matrices
        Wyw0 = softplus(p['yw0'])
        Wyw1 = softplus(p['yw1'])
        Wtw0 = softplus(p['tw0'])
        Wtw1 = softplus(p['tw1'])
        Wxy  = softplus(p['xw0_xy'])
        Wxt  = softplus(p['xw0_xt'])
        Wxw1 = softplus(p['xw1'])

        # y branch — softplus activation
        y_pre1 = y0 @ Wyw0 + p['yb0']
        y1     = softplus(y_pre1)
        y_pre2 = y1 @ Wyw1 + p['yb1']
        y2     = softplus(y_pre2)

        # z branch — sigmoid activation
        z_pre1 = z0 @ p['zw0'] + p['zb0']
        z1     = sigmoid(z_pre1)
        z_pre2 = z1 @ p['zw1'] + p['zb1']
        z2     = sigmoid(z_pre2)

        # t branch — sigmoid activation, non-neg weights
        t_pre1 = t0 @ Wtw0 + p['tb0']
        t1     = sigmoid(t_pre1)
        t_pre2 = t1 @ Wtw1 + p['tb1']
        t2     = sigmoid(t_pre2)

        # x layer 0 (eq. 4)
        x_pre1 = (x0 @ p['xw0_xx'] +
                  y2 @ Wxy +
                  z2 @ p['xw0_xz'] +
                  t2 @ Wxt +
                  p['xb0'])
        x1 = softplus(x_pre1)

        # x layer 1 (eq. 5, h=1), non-neg W_xx
        x_pre2 = x1 @ Wxw1 + p['xb1']
        x2     = softplus(x_pre2)

        out = x2 @ p['w_out'] + p['b_out']

        self.cache = dict(
            x0=x0, y0=y0, t0=t0, z0=z0,
            Wyw0=Wyw0, Wyw1=Wyw1, Wtw0=Wtw0, Wtw1=Wtw1,
            Wxy=Wxy, Wxt=Wxt, Wxw1=Wxw1,
            y_pre1=y_pre1, y1=y1, y_pre2=y_pre2, y2=y2,
            z_pre1=z_pre1, z1=z1, z_pre2=z_pre2, z2=z2,
            t_pre1=t_pre1, t1=t1, t_pre2=t_pre2, t2=t2,
            x_pre1=x_pre1, x1=x1, x_pre2=x_pre2, x2=x2,
        )
        return out

    def backward_from_cache(self, out, y_true):
        p = self.params
        c = self.cache
        N = y_true.shape[0]
        grads = {}

        # dL/d_out
        d_out = 2.0 * (out - y_true) / N        # shape (N, 1)

        # ── Output layer ──────────────────────────────────────────
        grads['w_out'] = c['x2'].T @ d_out       # (n, 1)
        grads['b_out'] = d_out.sum(axis=0)        # (1,)
        d_x2 = d_out @ p['w_out'].T              # (N, n)

        # ── x layer 1 ─────────────────────────────────────────────
        # x2 = softplus(x_pre2),  x_pre2 = x1 @ Wxw1 + xb1
        d_x_pre2 = d_x2 * softplus_grad(c['x_pre2'])   # (N, n)
        grads['xb1'] = d_x_pre2.sum(axis=0)

        # xw1 is constrained: W_actual = softplus(xw1_raw)
        # dL/d(raw) = dL/d(W_actual) * softplus_grad(raw)
        d_Wxw1 = c['x1'].T @ d_x_pre2                  # (n, n)
        grads['xw1'] = d_Wxw1 * softplus_grad(p['xw1'])

        d_x1 = d_x_pre2 @ c['Wxw1'].T                  # (N, n)

        # ── x layer 0 ─────────────────────────────────────────────
        # x1 = softplus(x_pre1)
        d_x_pre1 = d_x1 * softplus_grad(c['x_pre1'])   # (N, n)
        grads['xb0'] = d_x_pre1.sum(axis=0)

        # xw0_xx is free (no constraint at h=0)
        grads['xw0_xx'] = c['x0'].T @ d_x_pre1         # (1, n)

        # xw0_xy is constrained
        d_Wxy = c['y2'].T @ d_x_pre1                   # (n, n)
        grads['xw0_xy'] = d_Wxy * softplus_grad(p['xw0_xy'])

        # xw0_xz is free
        grads['xw0_xz'] = c['z2'].T @ d_x_pre1         # (n, n)

        # xw0_xt is constrained
        d_Wxt = c['t2'].T @ d_x_pre1                   # (n, n)
        grads['xw0_xt'] = d_Wxt * softplus_grad(p['xw0_xt'])

        # Propagate gradients back into branch final layers
        d_y2 = d_x_pre1 @ c['Wxy'].T                   # (N, n)
        d_z2 = d_x_pre1 @ p['xw0_xz'].T                # (N, n)
        d_t2 = d_x_pre1 @ c['Wxt'].T                   # (N, n)

        # ── y branch (backward through 2 layers) ──────────────────
        d_y_pre2 = d_y2 * softplus_grad(c['y_pre2'])   # (N, n)
        grads['yb1'] = d_y_pre2.sum(axis=0)
        d_Wyw1 = c['y1'].T @ d_y_pre2
        grads['yw1'] = d_Wyw1 * softplus_grad(p['yw1'])
        d_y1 = d_y_pre2 @ c['Wyw1'].T

        d_y_pre1 = d_y1 * softplus_grad(c['y_pre1'])
        grads['yb0'] = d_y_pre1.sum(axis=0)
        d_Wyw0 = c['y0'].T @ d_y_pre1
        grads['yw0'] = d_Wyw0 * softplus_grad(p['yw0'])

        # ── z branch (backward through 2 layers) ──────────────────
        d_z_pre2 = d_z2 * sigmoid_grad(c['z_pre2'])
        grads['zb1'] = d_z_pre2.sum(axis=0)
        grads['zw1'] = c['z1'].T @ d_z_pre2             # free weights
        d_z1 = d_z_pre2 @ p['zw1'].T

        d_z_pre1 = d_z1 * sigmoid_grad(c['z_pre1'])
        grads['zb0'] = d_z_pre1.sum(axis=0)
        grads['zw0'] = c['z0'].T @ d_z_pre1             # free

        # ── t branch (backward through 2 layers) ──────────────────
        d_t_pre2 = d_t2 * sigmoid_grad(c['t_pre2'])
        grads['tb1'] = d_t_pre2.sum(axis=0)
        d_Wtw1 = c['t1'].T @ d_t_pre2
        grads['tw1'] = d_Wtw1 * softplus_grad(p['tw1'])
        d_t1 = d_t_pre2 @ c['Wtw1'].T

        d_t_pre1 = d_t1 * sigmoid_grad(c['t_pre1'])
        grads['tb0'] = d_t_pre1.sum(axis=0)
        d_Wtw0 = c['t0'].T @ d_t_pre1
        grads['tw0'] = d_Wtw0 * softplus_grad(p['tw0'])

        return grads

    def predict(self, X):
        return self.forward(X)


class ISNN2_NumPy:
    """
    ISNN-2 with skip connections, implemented manually.
    Each x layer receives the current x hidden state, the original x0,
    and the current hidden states of all branches.
    Architecture: 1 hidden layer per branch, 15 neurons.
    """

    def __init__(self, n=15, seed=42):
        rng = np.random.RandomState(seed)
        s = 0.1

        self.params = {
            # y branch (1 hidden layer, non-neg weights)
            'yw0': rng.randn(1, n) * s,
            'yb0': np.zeros(n),

            # z branch (1 hidden layer, free)
            'zw0': rng.randn(1, n) * s,
            'zb0': np.zeros(n),

            # t branch (1 hidden layer, non-neg weights)
            'tw0': rng.randn(1, n) * s,
            'tb0': np.zeros(n),

            # x layer 0 — takes raw inputs (eq. 9)
            'xw0_xx': rng.randn(1, n) * s,   # free (h=0)
            'xw0_xy': rng.randn(1, n) * s,   # non-neg
            'xw0_xz': rng.randn(1, n) * s,   # free
            'xw0_xt': rng.randn(1, n) * s,   # non-neg
            'xb0':    np.zeros(n),

            # x layer 1 — skip connections (eq. 10)
            'xw1_xx':  rng.randn(n, n) * s,   # non-neg (h=1)
            'xw1_xx0': rng.randn(1, n) * s,   # skip from original x0
            'xw1_xy':  rng.randn(n, n) * s,   # non-neg
            'xw1_xz':  rng.randn(n, n) * s,   # free
            'xw1_xt':  rng.randn(n, n) * s,   # non-neg
            'xb1':     np.zeros(n),

            # output
            'w_out': rng.randn(n, 1) * s,
            'b_out': np.zeros(1),
        }
        self.adam = AdamState(self.params)
        self.cache = {}

    def forward(self, X):
        p = self.params
        x0 = X[:, 0:1]
        y0 = X[:, 1:2]
        t0 = X[:, 2:3]
        z0 = X[:, 3:4]

        # Constrained actual weights
        Wyw0    = softplus(p['yw0'])
        Wtw0    = softplus(p['tw0'])
        Wx0_xy  = softplus(p['xw0_xy'])
        Wx0_xt  = softplus(p['xw0_xt'])
        Wx1_xx  = softplus(p['xw1_xx'])
        Wx1_xy  = softplus(p['xw1_xy'])
        Wx1_xt  = softplus(p['xw1_xt'])

        # Branch hidden layers (from raw inputs)
        y_pre1 = y0 @ Wyw0 + p['yb0']
        y1     = softplus(y_pre1)

        z_pre1 = z0 @ p['zw0'] + p['zb0']
        z1     = sigmoid(z_pre1)

        t_pre1 = t0 @ Wtw0 + p['tb0']
        t1     = sigmoid(t_pre1)

        # x layer 0 — raw inputs only (eq. 9)
        x_pre1 = (x0 @ p['xw0_xx'] +
                  y0 @ Wx0_xy +
                  z0 @ p['xw0_xz'] +
                  t0 @ Wx0_xt +
                  p['xb0'])
        x1 = softplus(x_pre1)

        # x layer 1 — skip connections (eq. 10)
        x_pre2 = (x1 @ Wx1_xx +
                  x0 @ p['xw1_xx0'] +
                  y1 @ Wx1_xy +
                  z1 @ p['xw1_xz'] +
                  t1 @ Wx1_xt +
                  p['xb1'])
        x2 = softplus(x_pre2)

        out = x2 @ p['w_out'] + p['b_out']

        self.cache = dict(
            x0=x0, y0=y0, t0=t0, z0=z0,
            Wyw0=Wyw0, Wtw0=Wtw0,
            Wx0_xy=Wx0_xy, Wx0_xt=Wx0_xt,
            Wx1_xx=Wx1_xx, Wx1_xy=Wx1_xy, Wx1_xt=Wx1_xt,
            y_pre1=y_pre1, y1=y1,
            z_pre1=z_pre1, z1=z1,
            t_pre1=t_pre1, t1=t1,
            x_pre1=x_pre1, x1=x1,
            x_pre2=x_pre2, x2=x2,
        )
        return out

    def backward_from_cache(self, out, y_true):
        """Manual backprop for ISNN-2."""
        p = self.params
        c = self.cache
        N = y_true.shape[0]
        grads = {}

        # dL/d_out
        d_out = 2.0 * (out - y_true) / N       # (N, 1)

        # ── Output layer ──────────────────────────────────────────
        grads['w_out'] = c['x2'].T @ d_out      # (n, 1)
        grads['b_out'] = d_out.sum(axis=0)
        d_x2 = d_out @ p['w_out'].T             # (N, n)

        # ── x layer 1 (eq. 10) ────────────────────────────────────
        # x2 = softplus(x_pre2)
        d_x_pre2 = d_x2 * softplus_grad(c['x_pre2'])   # (N, n)
        grads['xb1'] = d_x_pre2.sum(axis=0)

        # xw1_xx (constrained, h=1)
        d_Wx1_xx = c['x1'].T @ d_x_pre2
        grads['xw1_xx'] = d_Wx1_xx * softplus_grad(p['xw1_xx'])
        d_x1 = d_x_pre2 @ c['Wx1_xx'].T        # gradient into x1

        # xw1_xx0 (skip from x0, unconstrained)
        grads['xw1_xx0'] = c['x0'].T @ d_x_pre2
        # gradient flows back to x0 through skip — accumulated later

        # xw1_xy (constrained)
        d_Wx1_xy = c['y1'].T @ d_x_pre2
        grads['xw1_xy'] = d_Wx1_xy * softplus_grad(p['xw1_xy'])
        d_y1 = d_x_pre2 @ c['Wx1_xy'].T

        # xw1_xz (free)
        grads['xw1_xz'] = c['z1'].T @ d_x_pre2
        d_z1 = d_x_pre2 @ p['xw1_xz'].T

        # xw1_xt (constrained)
        d_Wx1_xt = c['t1'].T @ d_x_pre2
        grads['xw1_xt'] = d_Wx1_xt * softplus_grad(p['xw1_xt'])
        d_t1 = d_x_pre2 @ c['Wx1_xt'].T

        # ── x layer 0 (eq. 9) ────────────────────────────────────
        # x1 = softplus(x_pre1)
        d_x_pre1 = d_x1 * softplus_grad(c['x_pre1'])   # (N, n)
        grads['xb0'] = d_x_pre1.sum(axis=0)

        # xw0_xx (free at h=0)
        grads['xw0_xx'] = c['x0'].T @ d_x_pre1

        # xw0_xy (constrained, connects raw y0 to x)
        d_Wx0_xy = c['y0'].T @ d_x_pre1
        grads['xw0_xy'] = d_Wx0_xy * softplus_grad(p['xw0_xy'])

        # xw0_xz (free)
        grads['xw0_xz'] = c['z0'].T @ d_x_pre1

        # xw0_xt (constrained)
        d_Wx0_xt = c['t0'].T @ d_x_pre1
        grads['xw0_xt'] = d_Wx0_xt * softplus_grad(p['xw0_xt'])

        # ── y branch backward ──────────────────────────────────────
        # y1 = softplus(y_pre1),  y_pre1 = y0 @ Wyw0 + yb0
        d_y_pre1 = d_y1 * softplus_grad(c['y_pre1'])
        grads['yb0'] = d_y_pre1.sum(axis=0)
        d_Wyw0 = c['y0'].T @ d_y_pre1
        grads['yw0'] = d_Wyw0 * softplus_grad(p['yw0'])

        # ── z branch backward ──────────────────────────────────────
        d_z_pre1 = d_z1 * sigmoid_grad(c['z_pre1'])
        grads['zb0'] = d_z_pre1.sum(axis=0)
        grads['zw0'] = c['z0'].T @ d_z_pre1     # free

        # ── t branch backward ──────────────────────────────────────
        d_t_pre1 = d_t1 * sigmoid_grad(c['t_pre1'])
        grads['tb0'] = d_t_pre1.sum(axis=0)
        d_Wtw0 = c['t0'].T @ d_t_pre1
        grads['tw0'] = d_Wtw0 * softplus_grad(p['tw0'])

        return grads

    def predict(self, X):
        return self.forward(X)


def train_numpy(model, X_train, y_train, X_test, y_test,
                epochs=5000, lr=1e-3):
    """Training loop for NumPy models using manual backprop and Adam."""
    model.adam.lr = lr
    ytr = y_train.reshape(-1, 1)
    yte = y_test.reshape(-1, 1)

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        # Forward pass (caches activations)
        out = model.forward(X_train)

        # Compute MSE loss
        train_loss = float(np.mean((out - ytr) ** 2))

        # Backward pass (manual backprop)
        grads = model.backward_from_cache(out, ytr)

        # Adam update
        model.adam.step(model.params, grads)

        # Test loss (no grad needed)
        test_out  = model.predict(X_test)
        test_loss = float(np.mean((test_out - yte) ** 2))

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % 1000 == 0:
            print(f"  epoch {epoch+1:5d} | train MSE: {train_loss:.6f} | test MSE: {test_loss:.6f}")

    return train_losses, test_losses


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_loss_curves(results, problem_num):
    """
    results = dict with keys like 'ISNN-1 PT', 'ISNN-2 PT', etc.
    Each value is (train_losses, test_losses).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Toy Problem {problem_num} — Training and Test MSE Loss', fontsize=13)

    styles = {
        'ISNN-1 (PyTorch)': ('blue',   '-'),
        'ISNN-2 (PyTorch)': ('orange', '-'),
        'ISNN-1 (NumPy)':   ('blue',   '--'),
        'ISNN-2 (NumPy)':   ('orange', '--'),
    }

    for name, (tr_loss, te_loss) in results.items():
        color, ls = styles.get(name, ('gray', '-'))
        axes[0].semilogy(tr_loss, label=name, color=color, linestyle=ls, linewidth=1.5)
        axes[1].semilogy(te_loss, label=name, color=color, linestyle=ls, linewidth=1.5)

    for ax, title in zip(axes, ['Training Loss (MSE)', 'Test Loss (MSE)']):
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE Loss')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'loss_curves_problem{problem_num}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: loss_curves_problem{problem_num}.png")


def plot_behavioral_response(models_pt, models_np, problem_num, test_high):
    """
    Vary each input from 0 to test_high while fixing the others at 2.0.
    Shows ground truth vs model predictions.

    models_pt = {'ISNN-1': model, 'ISNN-2': model}  (PyTorch)
    models_np = {'ISNN-1': model, 'ISNN-2': model}  (NumPy)
    """
    n_pts  = 200
    fixed  = 2.0
    fn     = toy_problem_1 if problem_num == 1 else toy_problem_2
    labels = ['x', 'y', 't', 'z']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Toy Problem {problem_num} — Behavioral Response of Trained Models', fontsize=13)

    for col, (var_idx, var_name) in enumerate(zip(range(4), labels)):
        ax = axes[col]
        pts = np.linspace(0.0, test_high, n_pts).astype(np.float32)

        # Build input array: vary one, fix the rest at 2.0
        X_scan = np.full((n_pts, 4), fixed, dtype=np.float32)
        X_scan[:, var_idx] = pts

        # Ground truth
        x_col, y_col, t_col, z_col = X_scan.T
        if problem_num == 1:
            truth = toy_problem_1(x_col, y_col, t_col, z_col)
        else:
            truth = toy_problem_2(x_col, y_col, z_col, t_col)

        ax.plot(pts, truth, 'k-', linewidth=2, label='True function')

        # PyTorch predictions
        X_t = torch.from_numpy(X_scan)
        for name, model in models_pt.items():
            model.eval()
            with torch.no_grad():
                pred = model(X_t).numpy().flatten()
            ax.plot(pts, pred, label=f'{name} (PT)', linewidth=1.5)

        # NumPy predictions
        for name, model in models_np.items():
            pred = model.predict(X_scan).flatten()
            ax.plot(pts, pred, linestyle='--', label=f'{name} (NP)', linewidth=1.5)

        ax.set_xlabel(var_name)
        ax.set_ylabel('f')
        ax.set_title(f'Varying {var_name}, others fixed at {fixed}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'behavioral_response_problem{problem_num}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: behavioral_response_problem{problem_num}.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_problem(problem_num, epochs=5000, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"  TOY PROBLEM {problem_num}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test = generate_dataset(problem_num)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Save datasets
    header = 'x,y,t,z,f'
    train_data = np.column_stack([X_train, y_train])
    test_data  = np.column_stack([X_test,  y_test])
    np.savetxt(f'dataset{problem_num}_train.csv', train_data, delimiter=',', header=header, comments='')
    np.savetxt(f'dataset{problem_num}_test.csv',  test_data,  delimiter=',', header=header, comments='')
    print(f"  Saved datasets: dataset{problem_num}_train.csv, dataset{problem_num}_test.csv")

    test_high = 6.0 if problem_num == 1 else 10.0
    results = {}

    # ── PyTorch ISNN-1
    print("\n[PyTorch] Training ISNN-1 ...")
    isnn1_pt = ISNN1_PyTorch(n=10)
    tr1, te1 = train_pytorch(isnn1_pt, X_train, y_train, X_test, y_test, epochs=epochs, lr=lr)
    results['ISNN-1 (PyTorch)'] = (tr1, te1)

    # ── PyTorch ISNN-2
    print("\n[PyTorch] Training ISNN-2 ...")
    isnn2_pt = ISNN2_PyTorch(n=15)
    tr2, te2 = train_pytorch(isnn2_pt, X_train, y_train, X_test, y_test, epochs=epochs, lr=lr)
    results['ISNN-2 (PyTorch)'] = (tr2, te2)

    # ── NumPy ISNN-1
    print("\n[NumPy] Training ISNN-1 ...")
    isnn1_np = ISNN1_NumPy(n=10, seed=42)
    isnn1_np.adam = AdamState(isnn1_np.params, lr=lr)
    tr3, te3 = train_numpy(isnn1_np, X_train, y_train, X_test, y_test, epochs=epochs, lr=lr)
    results['ISNN-1 (NumPy)'] = (tr3, te3)

    # ── NumPy ISNN-2
    print("\n[NumPy] Training ISNN-2 ...")
    isnn2_np = ISNN2_NumPy(n=15, seed=42)
    isnn2_np.adam = AdamState(isnn2_np.params, lr=lr)
    tr4, te4 = train_numpy(isnn2_np, X_train, y_train, X_test, y_test, epochs=epochs, lr=lr)
    results['ISNN-2 (NumPy)'] = (tr4, te4)

    # ── Plots
    plot_loss_curves(results, problem_num)

    models_pt = {'ISNN-1': isnn1_pt, 'ISNN-2': isnn2_pt}
    models_np = {'ISNN-1': isnn1_np, 'ISNN-2': isnn2_np}
    plot_behavioral_response(models_pt, models_np, problem_num, test_high)

    # Final metrics
    print(f"\n  Final test MSE (Problem {problem_num}):")
    for name, (tr, te) in results.items():
        print(f"    {name:25s}: train={tr[-1]:.6f}  test={te[-1]:.6f}")


if __name__ == '__main__':
    EPOCHS = 5000
    LR     = 1e-3

    run_problem(1, epochs=EPOCHS, lr=LR)
    run_problem(2, epochs=EPOCHS, lr=LR)
