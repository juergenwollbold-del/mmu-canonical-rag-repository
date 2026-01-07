import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  MMU Zeeman splitting from k23 coupling
#  - Plot 1: Δω_Z(B) / Δω_Z(B_ref) vs magnetic field B
#  - Plot 2: Δω_Z(ε_G) / Δω_Z(0) vs gravimetric compression ε_G
# ============================================================

# ----------------------------
# Physical constants (SI)
# ----------------------------
e    = 1.602176634e-19      # C
eps0 = 8.8541878128e-12     # F/m
hbar = 1.054571817e-34      # J s
c    = 2.99792458e8         # m/s
G    = 6.67430e-11          # m^3 kg^-1 s^-2 (not directly needed here)

# ----------------------------
# MMU UR-cell parameters
# ----------------------------
# Bohr-like internal length a1 (can be replaced by your a_int)
a1 = 5.29e-11               # m

# Effective inertial factor (absorbed into stiffnesses)
m_eff = 1.0

def k2_of_a(a):
    """Electric (Coulomb) stiffness as function of a."""
    return 2 * e**2 / (4 * np.pi * eps0 * a**3)

def k3_of_a(a):
    """Spin–torsion stiffness as function of a."""
    return hbar * c / a**3

def k23_max_of_a(a, frac=0.3):
    """
    Maximal stable k23 at given a:
    k23^2 < k2*k3  ->  k23_max = frac * sqrt(k2*k3)
    """
    k2 = k2_of_a(a)
    k3 = k3_of_a(a)
    return frac * np.sqrt(k2 * k3)

# ----------------------------
# 1) Zeeman vs magnetic field B (flat space, a = a1)
# ----------------------------
a_flat = a1
k2_flat = k2_of_a(a_flat)
k3_flat = k3_of_a(a_flat)
k23_max_flat = k23_max_of_a(a_flat, frac=0.3)

B_ref = 10.0  # Tesla (reference field for normalisation)

def k23_eff_flat(B):
    """Effective k23(B) in flat space (a = a1)."""
    return k23_max_flat * (B / B_ref)

print("=== MMU Zeeman model (w2-w3 subsystem, flat space) ===")
print(f"a1          = {a1:.3e} m")
print(f"k2(a1)      = {k2_flat:.3e} N/m")
print(f"k3(a1)      = {k3_flat:.3e} N/m")
print(f"k23_max(a1) = {k23_max_flat:.3e} N/m")

B_vals = np.logspace(-3, 1, 300)   # 1 mT ... 10 T

omega_plus  = []
omega_minus = []
delta_omega = []

for B in B_vals:
    k23 = k23_eff_flat(B)

    # 2x2 K-matrix eigenvalues
    trace    = k2_flat + k3_flat
    det_term = (k2_flat - k3_flat)**2 + 4 * k23**2

    lam_plus  = 0.5 * (trace + np.sqrt(det_term))
    lam_minus = 0.5 * (trace - np.sqrt(det_term))

    lam_minus = max(lam_minus, 0.0)  # numerical safety

    w_plus  = np.sqrt(lam_plus / m_eff)
    w_minus = np.sqrt(lam_minus / m_eff)

    omega_plus.append(w_plus)
    omega_minus.append(w_minus)
    delta_omega.append(w_plus - w_minus)

omega_plus  = np.array(omega_plus)
omega_minus = np.array(omega_minus)
delta_omega = np.array(delta_omega)

idx_ref   = np.argmin(np.abs(B_vals - B_ref))
delta_ref = delta_omega[idx_ref]
delta_norm_B = delta_omega / delta_ref

print(f"Reference B_ref = {B_ref} T")
print(f"Delta_omega(B_ref) = {delta_ref:.3e} rad/s")

# ----------------------------
# 2) Zeeman vs gravimetric compression ε_G (a(G))
# ----------------------------
# We fix B = B_ref and vary a(G) = a1 * (1 - ε_G)
B_fixed = B_ref

eps_G_vals = np.logspace(-8, -3, 200)  # 10^-8 ... 10^-3
delta_omega_G = []

for eps_G in eps_G_vals:
    a_G  = a1 * (1.0 - eps_G)

    k2_G = k2_of_a(a_G)
    k3_G = k3_of_a(a_G)

    # keep same fractional cross-coupling (geometry),
    # but scaled with new k2,k3
    k23_max_G = k23_max_of_a(a_G, frac=0.3)
    k23_G     = k23_max_G * (B_fixed / B_ref)  # same B-scaling

    trace_G    = k2_G + k3_G
    det_term_G = (k2_G - k3_G)**2 + 4 * k23_G**2

    lam_plus_G  = 0.5 * (trace_G + np.sqrt(det_term_G))
    lam_minus_G = 0.5 * (trace_G - np.sqrt(det_term_G))

    lam_minus_G = max(lam_minus_G, 0.0)

    w_plus_G  = np.sqrt(lam_plus_G / m_eff)
    w_minus_G = np.sqrt(lam_minus_G / m_eff)

    delta_omega_G.append(w_plus_G - w_minus_G)

delta_omega_G = np.array(delta_omega_G)

# normalise to flat-space value ε_G -> 0
delta_ref_G = delta_omega_G[0]
delta_norm_G = delta_omega_G / delta_ref_G

print(f"Delta_omega(G=0) = {delta_ref_G:.3e} rad/s")
print("Expected scaling ~ (a/a0)^(-3/2) ≈ 1 + (3/2) ε_G for small ε_G")

# ----------------------------
# Plot 1: Zeeman vs B
# ----------------------------
plt.figure(figsize=(7,5))
plt.loglog(B_vals, delta_norm_B)
plt.xlabel("Magnetic field B [T]")
plt.ylabel(r"Zeeman splitting $\Delta\omega_Z / \Delta\omega_Z(B_{\rm ref})$")
plt.title("MMU Zeeman splitting from k$_{23}$ coupling (vs B, flat space)")
plt.grid(True, which="both", ls=":")
plt.tight_layout()

# ----------------------------
# Plot 2: Zeeman vs gravimetric compression ε_G
# ----------------------------
plt.figure(figsize=(7,5))
plt.loglog(eps_G_vals, delta_norm_G)
plt.xlabel(r"gravimetric compression $\varepsilon_G$")
plt.ylabel(r"$\Delta\omega_Z(\varepsilon_G) / \Delta\omega_Z(0)$")
plt.title("MMU Zeeman response to gravimetric compression a(G)")
plt.grid(True, which="both", ls=":")
plt.tight_layout()

plt.show()
