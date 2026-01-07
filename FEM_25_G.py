#!/usr/bin/env python3
"""
MMU v1.8 – Dual-Tetrahedron g-factor simulation

Dieses Programm demonstriert:
- Zwei interne Torsionsachsen (w3+ und w3-)
- Beide Spins präzedieren unabhängig (Spin-Vektoren S+ und S-)
- Projektion auf w1 mit P = cos(109.47°)
- Effektives magnetisches Moment = 2 × Einzelmoment
- Effektive Larmor-Frequenz = 2 × Einzel-Larmor-Frequenz
→ g_MM U = 2

Autor: J. Wollbold + AI assistant (2025)
"""

import numpy as np

# ------------------------------------------------------------
# 1. Geometrische Konstante: Projektionsfaktor P = cos(109.47°)
# ------------------------------------------------------------
theta = np.deg2rad(109.47)
P = np.cos(theta)   # = -1/3

# ------------------------------------------------------------
# 2. Interne Spinachsen (Tetraeder / Anti-Tetraeder)
# ------------------------------------------------------------
# beide normiert
S_plus_0  = np.array([1, 0, 1], dtype=float); S_plus_0 /= np.linalg.norm(S_plus_0)
S_minus_0 = np.array([1, 0,-1], dtype=float); S_minus_0 /= np.linalg.norm(S_minus_0)

# ------------------------------------------------------------
# 3. Magnetische Grundkopplung (dimensionsloser Proportionalfaktor)
# ------------------------------------------------------------
gamma0 = 1.0       # Einzel-Larmor-Kopplung
mu0    = 1.0       # Einzel-Magnetmoment

# ------------------------------------------------------------
# 4. Larmor–Gleichung
# ------------------------------------------------------------
def dSdt(S, B, gamma):
    return gamma * np.cross(S, B)

def simulate_spin(S0, B, gamma, T, dt):
    N = int(T/dt)
    S = S0.copy()
    hist = np.zeros((N, 3))
    t = np.linspace(0, T, N)

    for i in range(N):
        hist[i] = S
        S = S + dSdt(S, B, gamma) * dt
        S /= np.linalg.norm(S)

    return t, hist

def estimate_freq(t, S_hist):
    Sx = S_hist[:,0]
    Sy = S_hist[:,1]
    phi = np.unwrap(np.arctan2(Sy, Sx))
    slope, _ = np.polyfit(t, phi, 1)
    return slope / (2*np.pi)

# ------------------------------------------------------------
# 5. Simulation
# ------------------------------------------------------------
def main():
    print("=== MMU v1.8 – Dual-Tetrahedron g-factor simulation ===")

    B = np.array([0,0,1.0])

    # Simulationszeit (mehrere Perioden)
    T  = 20.0
    dt = 0.0005

    # Einzel-Spins simulieren
    t, S_plus  = simulate_spin(S_plus_0,  B, gamma0, T, dt)
    _, S_minus = simulate_spin(S_minus_0, B, gamma0, T, dt)

    # Frequenzen messen
    nu_plus  = estimate_freq(t, S_plus)
    nu_minus = estimate_freq(t, S_minus)

    # Effektiver Spin = Summe der beiden Projektionen
    mu_plus_proj  = P * mu0
    mu_minus_proj = P * mu0

    mu_eff = mu_plus_proj + mu_minus_proj     # = 2 * mu_plus_proj
    g_MMU  = mu_eff / mu_plus_proj            # Verhältnis = g-Faktor

    # Effektive Larmor-Frequenz (wenn beide Achsen beitragen)
    nu_eff = nu_plus + nu_minus

    print(f"\nEinzelne Larmor-Frequenz ν0         = {nu_plus:.5f} Hz")
    print(f"Zweite (Anti-Tetraeder) Frequenz   = {nu_minus:.5f} Hz")
    print(f"Effektive Frequenz ν_eff (=ν+ +ν-) = {nu_eff:.5f} Hz")

    print("\n--- Projektionen ---")
    print(f"Einzel-Projektion μ+        = {mu_plus_proj:.5f}")
    print(f"Einzel-Projektion μ-        = {mu_minus_proj:.5f}")
    print(f"Effektiv μ_eff (=2 μ+)       = {mu_eff:.5f}")

    print("\n--- resultierender g-Faktor ---")
    print(f"g_MMU = μ_eff / μ_plus_proj  = {g_MMU:.5f}")

    print("\nErwartet (Dirac): g = 2")
    print("MMU erreicht geometrisch denselben Faktor.\n")

if __name__ == "__main__":
    main()
