import numpy as np
import matplotlib.pyplot as plt

# ============================================
#  Physical constants (SI)
# ============================================
hbar = 1.054_571_817e-34      # J·s
c    = 2.997_924_58e8         # m/s
G    = 6.674_30e-11           # m^3 kg^-1 s^-2
e    = 1.602_176_634e-19      # C
eps0 = 8.854_187_8128e-12     # F/m

m_e  = 9.109_383_7015e-31     # kg
m_p  = 1.672_621_923_69e-27   # kg
m_H  = m_e + m_p              # effective H mass for k4

alpha = e**2 / (4*np.pi*eps0*hbar*c)

# Bohr radius as a simple proxy for a1 (you can adjust to your MMU a1)
a1 = 5.291_772_109_03e-11     # m

# ============================================
#  MMU stiffness functions (SI units)
# ============================================
def k2(a):
    """Electric stiffness k2(a) = 2 e^2 / (4π ε0 a^3)"""
    return 2*e**2 / (4*np.pi*eps0 * a**3)

def k3(a):
    """Torsional/spin stiffness k3(a) = ħ c / a^3"""
    return hbar * c / a**3

def k4(a, m=m_H):
    """Volumetric/gravimetric stiffness k4(a) = 2 G m^2 / a^3"""
    return 2 * G * m**2 / a**3

# calibrated Lamb coupling prefactor from MMU4 (order of magnitude)
k34_0 = 1.1e-6

def k34(a, m=m_H):
    """Torsion–volume coupling k34 ∝ sqrt(k3 k4)"""
    return k34_0 * np.sqrt(k3(a) * k4(a, m))

def omega_Lamb_raw(a, m=m_H):
    """
    Raw Lamb 'frequency scale' ∝ k34^2 / |k3 - k4|.
    Units are arbitrary here; we only care about ratios.
    """
    k3_val = k3(a)
    k4_val = k4(a, m)
    k34_val = k34(a, m)
    denom = np.abs(k3_val - k4_val)
    return k34_val**2 / denom

# ============================================
#  Part 1: Lamb shift vs internal length a
#  (from Bohr scale down towards EG)
# ============================================
# Scan a from a1 down to, say, 1e-3 * a1
a_vals = np.logspace(np.log10(a1), np.log10(a1*1e-3), 300)
omega_vals = omega_Lamb_raw(a_vals)

# Normalise to value at a1 for easier comparison
omega_norm = omega_vals / omega_vals[0]

plt.figure()
plt.loglog(a_vals/a1, omega_norm)
plt.xlabel("a / a1")
plt.ylabel("ω_Lamb(a) / ω_Lamb(a1)")
plt.title("MMU Lamb-scale vs internal length a")
plt.grid(True, which="both")
plt.tight_layout()

# ============================================
#  Part 2: Lamb shift vs gravitational compression ε_G
#  for a fixed hydrogenic level n
# ============================================
n = 1  # choose n=1,2,... as you like
a_n = a1 / n**2

# Gravimetric compression parameter ε_G
eps_G = np.logspace(-8, -3, 200)  # from very weak to magnetar-like

a_G = a_n * (1 - eps_G)  # compressed internal length
omega_G = omega_Lamb_raw(a_G)
omega_G_norm = omega_G / omega_Lamb_raw(a_n)  # relative Lamb shift

plt.figure()
plt.loglog(eps_G, omega_G_norm - 1.0)
plt.xlabel("ε_G")
plt.ylabel("relative Lamb shift Δω_L / ω_L(0)")
plt.title("MMU Lamb-shift response to gravimetric compression")
plt.grid(True, which="both")
plt.tight_layout()

plt.show()
