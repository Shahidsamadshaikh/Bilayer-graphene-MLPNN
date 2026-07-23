"""
=============================================================================
GENERATE FERMI VELOCITY DATA FROM TRUSTED TB MODEL
=============================================================================
Computes Fermi velocity at the K‑point for all (V, ε) in the grid.
Output: fermi_velocity_data.csv (861 rows)
=============================================================================
"""

import numpy as np
from pythtb import tb_model
import pandas as pd
import time

# ---- Parameters (exactly as in effecint.py) ----
a = 2.46
d = 3.35
gamma0_0 = -2.70
gamma1_0 = 0.40
gamma3_0 = 0.30
gamma4_0 = 0.15
nu = 0.165
nu_zz = 0.33
beta_g0 = 3.37
beta_g1 = 4.00
beta_g3 = 5.00
beta_g4 = 5.00
bond_angles = [0., 120., 240.]

def inplane_delta(theta_deg, eps, nu):
    th = np.radians(theta_deg)
    return eps * (np.cos(th)**2 - nu * np.sin(th)**2)

def new_hop(t0, delta, beta):
    return t0 * np.exp(-beta * delta)

def compute_hoppings(eps):
    delta_intra = [inplane_delta(th, eps, nu) for th in bond_angles]
    g0_strained = [new_hop(gamma0_0, dd, beta_g0) for dd in delta_intra]
    delta_z = -nu_zz * eps
    g1_strained = new_hop(gamma1_0, delta_z, beta_g1)
    delta_oblique = 0.5 * np.mean(delta_intra) + 0.5 * (-nu_zz * eps * 0.5)
    g3_strained = new_hop(gamma3_0, delta_oblique, beta_g3)
    g4_strained = new_hop(gamma4_0, delta_oblique, beta_g4)
    return g0_strained, g1_strained, g3_strained, g4_strained

def build_blg(g0_list, g1, g3, g4, V_field):
    dz = d / 20.0
    lat = [[a, 0., 0.],
           [a/2., a*np.sqrt(3)/2., 0.],
           [0., 0., 20.]]
    orb = [[0., 0., 0.],
           [1/3., 2/3., 0.],
           [1/3., 2/3., dz],
           [0., 0., dz]]
    model = tb_model(2, 3, lat, orb)
    model.set_onsite([-V_field/2, -V_field/2, V_field/2, V_field/2])
    model.set_hop(g0_list[0], 0, 1, [ 0,  0, 0])
    model.set_hop(g0_list[1], 0, 1, [-1,  0, 0])
    model.set_hop(g0_list[2], 0, 1, [ 0, -1, 0])
    model.set_hop(g0_list[0], 2, 3, [ 0,  0, 0])
    model.set_hop(g0_list[1], 2, 3, [-1,  0, 0])
    model.set_hop(g0_list[2], 2, 3, [ 0, -1, 0])
    model.set_hop(g1, 1, 2, [0, 0, 0])
    model.set_hop(g3, 0, 3, [ 0,  0, 0])
    model.set_hop(g3, 0, 3, [ 1,  0, 0])
    model.set_hop(g3, 0, 3, [ 0,  1, 0])
    model.set_hop(g4, 0, 2, [ 0,  0, 0])
    model.set_hop(g4, 0, 2, [-1,  0, 0])
    model.set_hop(g4, 0, 2, [ 0, -1, 0])
    model.set_hop(g4, 1, 3, [ 0,  0, 0])
    model.set_hop(g4, 1, 3, [ 1,  0, 0])
    model.set_hop(g4, 1, 3, [ 0,  1, 0])
    return model

# ---- Function to compute Fermi velocity at K‑point ----
def compute_fermi_velocity(V, eps, delta_k=1e-6):
    """
    Returns (vx, vy) in m/s at the K‑point (1/3, 2/3) for the conduction band.
    Uses central finite differences.
    """
    kx0, ky0 = 1/3, 2/3

    # Build TB model for this (V, eps)
    g0, g1, g3, g4 = compute_hoppings(eps)
    model = build_blg(g0, g1, g3, g4, V)

    # Helper to get conduction band energy (sorted)
    def get_E(kx, ky):
        evals = model.solve_all([[kx, ky]])[:, 0]  # shape (4,)
        sorted_evals = np.sort(evals)              # sort to get physical order
        return sorted_evals[2]                     # conduction band (E3)

    E0 = get_E(kx0, ky0)
    E_x_plus = get_E(kx0 + delta_k, ky0)
    E_x_minus = get_E(kx0 - delta_k, ky0)
    E_y_plus = get_E(kx0, ky0 + delta_k)
    E_y_minus = get_E(kx0, ky0 - delta_k)

    dE_dkx = (E_x_plus - E_x_minus) / (2 * delta_k)
    dE_dky = (E_y_plus - E_y_minus) / (2 * delta_k)

    # Convert from fractional k to physical velocity
    # v = (1/ħ) * ∂E/∂k_physical, and k_physical = (2π/a) * k_frac
    # So ∂E/∂k_frac = (2π/a) * ∂E/∂k_physical => ∂E/∂k_physical = (a/(2π)) * ∂E/∂k_frac
    a_m = a * 1e-10                   # in meters
    hbar = 6.582119569e-16            # eV·s
    conv = a_m / (2 * np.pi * hbar)   # conversion factor

    vx = dE_dkx * conv
    vy = dE_dky * conv
    return vx, vy

# ---- Grid parameters (same as your CSV) ----
V_values = np.arange(0.1, 5.01, 0.12)
eps_values = np.linspace(-0.10, 0.10, 21)

# ---- Generate data ----
print("Generating Fermi velocity data...")
start = time.time()
results = []
total = len(V_values) * len(eps_values)
count = 0
for V in V_values:
    for eps in eps_values:
        vx, vy = compute_fermi_velocity(V, eps)
        v_mag = np.sqrt(vx**2 + vy**2)
        results.append([V, eps, vx, vy, v_mag])
        count += 1
        if count % 100 == 0:
            print(f"Progress: {count}/{total}")
elapsed = time.time() - start
print(f"Done in {elapsed:.1f} seconds.")

# ---- Save to CSV ----
df = pd.DataFrame(results, columns=['V (eV)', 'eps', 'vx (m/s)', 'vy (m/s)', 'v_mag (m/s)'])
df.to_csv("fermi_velocity_data.csv", index=False)
print("Data saved to 'fermi_velocity_data.csv'.")
print(f"Total points: {len(df)}")