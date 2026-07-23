"""
train_fermi_mlp.py
-------------------
Trains an MLP to predict the Fermi velocity vector (vx, vy) of strained,
gated bilayer graphene from tight-binding data (V, strain) -> (vx, vy).

Design choices (see accompanying paper discussion):
  * The network predicts vx, vy (not v_mag directly) and v_mag is computed
    as sqrt(vx^2 + vy^2) downstream. This makes a negative Fermi speed
    structurally impossible, unlike a direct v_mag regression.
  * The training loss is weighted by 1/(v_mag_true + eps) so that the
    low-velocity "valley" region (which carries the physics of interest -
    the critical field Vc(strain), and the strain=0 symmetry line) is not
    drowned out by the numerically much larger saturated (~1.2e6 m/s) bulk.
  * Inputs are standardized; outputs are scaled to O(1) for stable training.
  * A held-out test set is used for all reported metrics/plots (no peeking).

Outputs (written to ./outputs/):
  learning_curve.png       - train/val loss vs epoch
  parity_plot.png          - predicted vs true v_mag on the test set
  residual_plot.png        - residuals vs true v_mag on the test set
  phase_diagram_nn.png     - fine-grid NN-predicted phase diagram
  line_cuts.png            - v_mag vs V at fixed strain, NN vs tight-binding
  metrics.json             - all reported numbers, for reproducibility
  fermi_mlp.pt             - trained model weights + scalers
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

torch.manual_seed(0)
np.random.seed(0)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# 1. Data loading and splitting
# --------------------------------------------------------------------------
df = pd.read_csv("fermi_velocity_data.csv")
X = df[["V (eV)", "eps"]].to_numpy(dtype=np.float32)
Y = df[["vx (m/s)", "vy (m/s)"]].to_numpy(dtype=np.float32)
v_mag_true_all = np.sqrt((Y ** 2).sum(axis=1))

n = len(df)
idx = np.random.permutation(n)
n_test = int(0.15 * n)
n_val = int(0.15 * n)
test_idx = idx[:n_test]
val_idx = idx[n_test:n_test + n_val]
train_idx = idx[n_test + n_val:]

# --------------------------------------------------------------------------
# 2. Scaling (fit on TRAIN only, to avoid leakage)
# --------------------------------------------------------------------------
x_mean, x_std = X[train_idx].mean(0), X[train_idx].std(0)
Y_SCALE = 1.0e6  # ~ saturation velocity scale, keeps outputs O(1)


def scale_x(a):
    return (a - x_mean) / x_std


def scale_y(a):
    return a / Y_SCALE


def unscale_y(a):
    return a * Y_SCALE


Xs = scale_x(X)
Ys = scale_y(Y)

Xtr = torch.tensor(Xs[train_idx], device=DEVICE)
Ytr = torch.tensor(Ys[train_idx], device=DEVICE)
Xval = torch.tensor(Xs[val_idx], device=DEVICE)
Yval = torch.tensor(Ys[val_idx], device=DEVICE)
Xte = torch.tensor(Xs[test_idx], device=DEVICE)
Yte = torch.tensor(Ys[test_idx], device=DEVICE)

# sample weights: emphasize the low-velocity valley / strain=0 line so the
# saturated bulk (majority of points, ~1.2e6 m/s) doesn't dominate the loss
v_train_true = v_mag_true_all[train_idx]
w_train = torch.tensor(
    1.0 / (v_train_true / Y_SCALE + 0.05), device=DEVICE, dtype=torch.float32
).unsqueeze(1)
w_train = w_train / w_train.mean()  # keep loss scale comparable to unweighted MSE


# --------------------------------------------------------------------------
# 3. Model
# --------------------------------------------------------------------------
class FermiMLP(nn.Module):
    def __init__(self, hidden=(128, 128, 64)):
        super().__init__()
        layers = []
        in_dim = 2
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.SiLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 2)]  # outputs: vx, vy (scaled)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = FermiMLP().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=25
)

# --------------------------------------------------------------------------
# 4. Training loop with early stopping
# --------------------------------------------------------------------------
max_epochs = 3000
patience = 200
best_val = float("inf")
best_state = None
wait = 0
history = {"train_loss": [], "val_loss": []}

for epoch in range(max_epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(Xtr)
    per_sample_mse = ((pred - Ytr) ** 2).mean(dim=1, keepdim=True)
    loss = (per_sample_mse * w_train).mean()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(Xval)
        val_loss = ((val_pred - Yval) ** 2).mean()

    history["train_loss"].append(loss.item())
    history["val_loss"].append(val_loss.item())
    scheduler.step(val_loss)

    if val_loss.item() < best_val - 1e-9:
        best_val = val_loss.item()
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss {best_val:.3e})")
            break

model.load_state_dict(best_state)
model.eval()

# --------------------------------------------------------------------------
# 5. Evaluation on held-out test set
# --------------------------------------------------------------------------
with torch.no_grad():
    pred_test = unscale_y(model(Xte).cpu().numpy())
true_test = unscale_y(Yte.cpu().numpy())

vmag_pred = np.sqrt((pred_test ** 2).sum(axis=1))
vmag_true = np.sqrt((true_test ** 2).sum(axis=1))


def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def mae(true, pred):
    return np.mean(np.abs(true - pred))


def rmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))


low_mask = vmag_true < 0.3e6
metrics = {
    "n_train": int(len(train_idx)),
    "n_val": int(len(val_idx)),
    "n_test": int(len(test_idx)),
    "overall": {
        "R2": float(r2(vmag_true, vmag_pred)),
        "MAE": float(mae(vmag_true, vmag_pred)),
        "RMSE": float(rmse(vmag_true, vmag_pred)),
    },
    "low_v_subset_true_lt_0.3e6": {
        "n": int(low_mask.sum()),
        "R2": float(r2(vmag_true[low_mask], vmag_pred[low_mask])) if low_mask.sum() > 1 else None,
        "MAE": float(mae(vmag_true[low_mask], vmag_pred[low_mask])) if low_mask.sum() > 0 else None,
        "RMSE": float(rmse(vmag_true[low_mask], vmag_pred[low_mask])) if low_mask.sum() > 0 else None,
    },
    "high_v_subset_true_ge_0.3e6": {
        "n": int((~low_mask).sum()),
        "R2": float(r2(vmag_true[~low_mask], vmag_pred[~low_mask])),
        "MAE": float(mae(vmag_true[~low_mask], vmag_pred[~low_mask])),
        "RMSE": float(rmse(vmag_true[~low_mask], vmag_pred[~low_mask])),
    },
    "any_negative_predictions": bool((vmag_pred < 0).any()),  # structurally impossible now
}
with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))

torch.save(
    {"state_dict": model.state_dict(), "x_mean": x_mean, "x_std": x_std, "y_scale": Y_SCALE},
    OUT / "fermi_mlp.pt",
)

print("Training complete. Generating plots...")

# --------------------------------------------------------------------------
# 6. Plots
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 6a. Learning curve ---------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history["train_loss"], label="Training Loss", lw=1.5)
ax.plot(history["val_loss"], label="Validation Loss", lw=1.5)
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss (scaled)")
ax.set_title("Learning Curves ($\\varepsilon = 0$ included)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "learning_curve.png", dpi=150)
plt.close(fig)

# --- 6b. Parity plot --------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(vmag_true, vmag_pred, alpha=0.6, s=28, edgecolor="none")
lims = [0, max(vmag_true.max(), vmag_pred.max()) * 1.05]
ax.plot(lims, lims, "r--", lw=1.5, label="y = x")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("True v_mag (m/s)")
ax.set_ylabel("Predicted v_mag (m/s)")
ax.set_title("NN Test Performance (all $\\varepsilon$)")
txt = (f"$R^2$ = {metrics['overall']['R2']:.4f}\n"
       f"MAE = {metrics['overall']['MAE']:.2e} m/s\n"
       f"low-v $R^2$ (<0.3e6) = {metrics['low_v_subset_true_lt_0.3e6']['R2']:.3f}")
ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8))
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "parity_plot.png", dpi=150)
plt.close(fig)

# --- 6c. Residual plot ------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
residuals = vmag_pred - vmag_true
ax.scatter(vmag_true, residuals, alpha=0.6, s=28, edgecolor="none")
ax.axhline(0, color="r", ls="--", lw=1.5)
ax.set_xlabel("True v_mag (m/s)")
ax.set_ylabel("Residual: Predicted − True (m/s)")
ax.set_title("Residuals vs. True Fermi Velocity (test set)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "residual_plot.png", dpi=150)
plt.close(fig)

# --- 6d. Fine-grid NN phase diagram ------------------------------------
V_fine = np.linspace(0.05, 5.0, 300)
eps_fine = np.linspace(-0.1, 0.1, 240)
Vg, Eg = np.meshgrid(V_fine, eps_fine)
grid_X = np.stack([Vg.ravel(), Eg.ravel()], axis=1).astype(np.float32)
grid_Xs = scale_x(grid_X)
with torch.no_grad():
    grid_pred = unscale_y(model(torch.tensor(grid_Xs, device=DEVICE)).cpu().numpy())
grid_vmag = np.sqrt((grid_pred ** 2).sum(axis=1)).reshape(Vg.shape)
grid_vmag = np.clip(grid_vmag, 0, None)  # structurally near-zero already; guard fp noise

fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(Vg, Eg, grid_vmag, levels=60, cmap="viridis", vmin=0)
cbar = fig.colorbar(cf, ax=ax)
cbar.set_label("Fermi velocity (m/s)")
ax.set_xlabel("Electric field V (eV)")
ax.set_ylabel("Strain $\\varepsilon$")
ax.set_title("Fermi Velocity Phase Diagram (NN, $\\varepsilon$=0 included)")
fig.tight_layout()
fig.savefig(OUT / "phase_diagram_nn.png", dpi=150)
plt.close(fig)

# --- 6e. Line cuts: v_mag vs V at fixed strain, NN vs tight-binding ----
cut_strains = [-0.10, -0.05, 0.0, 0.05, 0.10]
fig, ax = plt.subplots(figsize=(8, 5.5))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(cut_strains)))
for c, eps_val in zip(colors, cut_strains):
    sub = df[np.isclose(df["eps"], eps_val, atol=1e-6)].sort_values("V (eV)")
    if len(sub):
        ax.scatter(sub["V (eV)"], sub["v_mag (m/s)"], color=c, s=18, alpha=0.7,
                   label=f"TB data, $\\varepsilon$={eps_val:+.2f}")
    Vq = np.linspace(0.05, 5.0, 400).astype(np.float32)
    Xq = scale_x(np.stack([Vq, np.full_like(Vq, eps_val)], axis=1))
    with torch.no_grad():
        Yq = unscale_y(model(torch.tensor(Xq, device=DEVICE)).cpu().numpy())
    vq = np.clip(np.sqrt((Yq ** 2).sum(axis=1)), 0, None)
    ax.plot(Vq, vq, color=c, lw=1.8)
ax.set_xlabel("Electric field V (eV)")
ax.set_ylabel("Fermi velocity (m/s)")
ax.set_title("NN prediction vs. tight-binding data (line cuts)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "line_cuts.png", dpi=150)
plt.close(fig)

print(f"All outputs written to: {OUT.resolve()}")