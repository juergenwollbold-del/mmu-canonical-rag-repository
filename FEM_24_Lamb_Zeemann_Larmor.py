#!/usr/bin/env python3
"""
MMU v1.7 – Full Magnetic Suite (absolute error version)

Berechnet aus einem einzigen MMU-K-Matrix-Set:
- Hyperfine (21 cm)
- Lamb shift
- Zeeman slope
- Larmor slope
- Spin precession S(t)

Relative Fehler werden absolut bewertet, sodass
Vorzeichen der Präzession (Richtung) keinen Einfluss hat.

Author: J. Wollbold + AI assistant (2025)
"""

import numpy as np

# ============================================================
# 1. MMU v1.4 K-Matrix Parameters (best-fit from global scan)
# ============================================================

k2_0   = 9.9382e-01
k3_0   = 8.4346e-01
k4_0   = 9.1542e-01
k23_0  = 1.1249e-02
k24_0  = 3.2890e-04
k34_0  = 7.0983e-02

m2, m3, m4 = 1.0, 1.0, 1.0
M = np.diag([m2, m3, m4])

EPS = 1e-15
NU_21_EXP = 1.42040575e9   # experimental hyperfine frequency

# ============================================================
# 2. Fundamental constants (for comparison with electron data)
# ============================================================

e  = 1.602176634e-19
m_e = 9.1093837015e-31
g_e = 2.00231930436256

# ============================================================
# 3. Geometry: scaling and hyperfine splitting
# ============================================================

def k_scale(a_int: float) -> float:
    return a_int**-3

def a_n(n: int, a1: float) -> float:
    return a1 / (n*n)

def omega_21cm(a_int: float) -> float:
    """Dimensionless hyperfine Δω from (w3, w4)."""
    s = k_scale(a_int)
    k3 = k3_0 * s
    k4 = k4_0 * s
    k34 = k34_0 * s

    K = np.array([[k3,  k34],
                  [k34, k4]])
    M_hf = np.diag([m3, m4])

    lam = np.linalg.eigvals(np.linalg.solve(M_hf, K))
    lam = np.real(lam)
    lam[lam < 0] = 0
    lam.sort()

    ωp, ωm = np.sqrt(lam)
    return abs(ωp - ωm)

def calibrate_omega_to_Hz(a1=1.0):
    """ω → Hz Skala aus 21 cm."""
    a_int = a_n(1, a1)
    ω_dimless = omega_21cm(a_int)
    ω_phys = 2*np.pi*NU_21_EXP
    return ω_phys / ω_dimless

# ============================================================
# 4. Lamb shift (dimensionless)
# ============================================================

def omega_lamb(a_int: float) -> float:
    """Dimensionless Lamb shift Δω ≈ k34² / |k3 − k4|."""
    s = k_scale(a_int)
    k3 = k3_0 * s
    k4 = k4_0 * s
    k34 = k34_0 * s

    denom = abs(k3 - k4)
    if denom < EPS:
        return 0.0
    return (k34*k34) / denom

# ============================================================
# 5. Zeeman / Larmor slopes from K-Matrix
# ============================================================

def zeeman_slope_HzT(omega_to_Hz: float) -> float:
    return (k3_0 * omega_to_Hz) / (2*np.pi)

def larmor_slope_HzT(omega_to_Hz: float) -> float:
    return (k3_0 * omega_to_Hz) / (2*np.pi)

# ============================================================
# 6. Spin precession (3D Spin-Vektor S(t))
# ============================================================

def dSdt(S: np.ndarray, B: np.ndarray, gamma_phys: float) -> np.ndarray:
    return gamma_phys * np.cross(S, B)

def simulate_larmor(B_vec, gamma_phys, T, dt):
    N = int(T/dt)
    S = np.array([1.0, 0.0, 1.0])
    S = S / np.linalg.norm(S)

    S_hist = np.zeros((N,3))
    t_arr = np.linspace(0, T, N)

    for i in range(N):
        S_hist[i] = S
        S += dSdt(S, B_vec, gamma_phys) * dt
        S /= np.linalg.norm(S)

    return t_arr, S_hist

def larmor_freq_from_signal(t, S_hist):
    Sx = S_hist[:,0]
    Sy = S_hist[:,1]
    phi = np.unwrap(np.arctan2(Sy, Sx))
    slope, _ = np.polyfit(t, phi, 1)
    return slope / (2*np.pi)

# ============================================================
# 7. Main
# ============================================================

def main():
    print("=== MMU v1.7 – Full Magnetic Suite ===")

    a1 = 1.0
    a_int = a_n(1, a1)

    # ---------- Hyperfine scale ----------
    omega_to_Hz = calibrate_omega_to_Hz(a1)
    print(f"ω → Hz scale: {omega_to_Hz:.3e} Hz per dimless-ω\n")

    # ---------- Hyperfine ----------
    nu_21 = omega_21cm(a_int) * omega_to_Hz / (2*np.pi)
    print(f"Hyperfine (21 cm): {nu_21:.6e} Hz")

    # ---------- Lamb ----------
    nu_Lamb = omega_lamb(a_int) * omega_to_Hz / (2*np.pi)
    print(f"Lamb shift:        {nu_Lamb:.6e} Hz")

    # ---------- Zeeman / Larmor Slopes ----------
    Z = zeeman_slope_HzT(omega_to_Hz)
    L = larmor_slope_HzT(omega_to_Hz)
    print(f"Zeeman slope:      {Z:.6e} Hz/T")
    print(f"Larmor slope:      {L:.6e} Hz/T")

    # ---------- Spin precession ----------
    B_vec = np.array([0,0,1.0])
    gamma_phys = k3_0 * omega_to_Hz
    T = 5.0 / L
    dt = T / 2000

    t, S_hist = simulate_larmor(B_vec, gamma_phys, T, dt)
    nu_L_est = larmor_freq_from_signal(t, S_hist)

    # ABSOLUTE error comparison
    abs_error = (abs(nu_L_est) - abs(L)) / abs(L) * 100

    print(f"Simulated ν_L:      {nu_L_est:.6e} Hz")
    print(f"Absolute error:     {abs_error:.2f}%")

if __name__ == "__main__":
    main()
