#!/usr/bin/env python3
"""
MMU v2.0 – dynamic g-2 simulation (C3)

Wir simulieren:
- Spinpräzession S(t) der beiden internen Achsen (w3+ und w3-)
- einmal mit k34 = 0
- einmal mit k34 = full (MMU value)
→ Extraktion der Larmorfrequenzen
→ Bestimmung von g0, g1
→ Δg = g1 − g0

Autor: J. Wollbold + AI assistant (2025)
"""

import numpy as np

# ==========================================
# 1. K-Matrix Basisparameter (MMU v1.4)
# ==========================================
k3  = 0.84346
k4  = 0.91542
k34 = 0.070983     # real coupling
k34_off = 0.0      # for comparison

# geometrischer Larmor-Basisterm (dimensionslos)
gamma0 = k3

# ==========================================
# 2. Spin-Vektoren (Tetraeder + Anti-Tetraeder)
# ==========================================
S_plus_0  = np.array([1.0, 0.0, 1.0], dtype=float)
S_plus_0  /= np.linalg.norm(S_plus_0)

S_minus_0 = np.array([1.0, 0.0,-1.0], dtype=float)
S_minus_0 /= np.linalg.norm(S_minus_0)

def dSdt(S, B, gamma):
    return gamma * np.cross(S, B)

def simulate_spin(S0, gamma, B, T, dt):
    N = int(T/dt)
    S = S0.copy()
    hist = np.zeros((N, 3))
    t = np.linspace(0, T, N)
    for i in range(N):
        hist[i] = S
        S += dSdt(S, B, gamma) * dt
        S /= np.linalg.norm(S)
    return t, hist

def extract_freq(t, S_hist):
    Sx, Sy = S_hist[:,0], S_hist[:,1]
    phi = np.unwrap(np.arctan2(Sy, Sx))
    slope, _ = np.polyfit(t, phi, 1)
    return slope / (2*np.pi)

# ==========================================
# 3. Effektive γ(k34)
# ==========================================
def gamma_eff(k3, k4, k34):
    # Geometrisch hergeleiteter Faktor:
    eta = (1/6) * (k3 / (k3 + k4))

    return k3 + eta * (k34**2 / abs(k3 - k4))

# ==========================================
# 4. g-Faktor Simulation
# ==========================================
def simulate_g(k3, k4, k34):
    B = np.array([0,0,1])
    T = 200
    dt = 0.002

    # gamma ohne Kopplung
    g0_plus  = gamma_eff(k3, k4, 0.0)
    g0_minus = gamma_eff(k3, k4, 0.0)

    # gamma mit Kopplung
    g1_plus  = gamma_eff(k3, k4, k34)
    g1_minus = gamma_eff(k3, k4, k34)

    # Jede Spinachse separat simulieren:
    t0p, S0p = simulate_spin(S_plus_0,  g0_plus,  B, T, dt)
    t0m, S0m = simulate_spin(S_minus_0, g0_minus, B, T, dt)

    t1p, S1p = simulate_spin(S_plus_0,  g1_plus,  B, T, dt)
    t1m, S1m = simulate_spin(S_minus_0, g1_minus, B, T, dt)

    # Frequenzen extrahieren:
    f0 = extract_freq(t0p, S0p) + extract_freq(t0m, S0m)
    f1 = extract_freq(t1p, S1p) + extract_freq(t1m, S1m)

    # g-Werte
    g0 = f0 / f0   # = 1 (normiert)
    g1 = f1 / f0

    # Δg
    delta_g = g1 - g0

    return delta_g, f0, f1

# ==========================================
# 5. Main
# ==========================================
def main():
    print("=== MMU v2.0 – dynamic g-2 Simulation ===")

    dg, f0, f1 = simulate_g(k3, k4, k34)

    print(f"Basis-Frequenz f0 (k34=0): {f0:.8f} Hz")
    print(f"Full  Frequenz f1 (k34):   {f1:.8f} Hz")
    print(f"Δg(MMU dynamisch)        = {dg:.8f}")

    # Vergleich
    dg_exp = 0.00231930436256
    g_exp  = 2.00231930436256

    print(f"\nExperiment Δg(electron)  = {dg_exp:.8f}")
    print(f"Relativer Fehler         = {(dg - dg_exp)/dg_exp * 100:.3f}%")

if __name__ == "__main__":
    main()
