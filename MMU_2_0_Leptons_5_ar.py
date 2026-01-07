"""
MMU 2.0 – Lepton benchmark (clean version)
=========================================

This script computes and compares four key atomic observables
for lepton families (electron, muon, tau) in a transparent way:

1. Balmer-α line (3→2) for e, µ, τ hydrogen:
   - Energy scales with the reduced mass µ_ell.
   - This reproduces standard QM and the MMU k2(a) / Rydberg derivation.

2. Lamb shift (2S–2P), MMU-effective m^2 model:
   - Based on the MMU hybrid picture:
       w3 = lepton torsion mode,
       w4 = volumetric mode (mostly nuclear),
       Lamb ∝ coupling of w3 to w4.
   - Effective scaling:
       ΔE_Lamb(m_l) ∝ m_l^2
     calibrated on e-H, then predicting µ-H and τ-H.

3. Hyperfine (21 cm-like):
   - Hyperfine splitting is torsional (w3–w4) in MMU.
   - Simple mass-based scaling:
       ν_hfs(m_l) = ν_21_H * (µ_l / µ_e)^3

4. Zeeman / Larmor:
   - Larmor frequency per Tesla:
       ν_L = g q / (4π m)
   - Computed for e, µ, τ.

This is an *effective benchmark script*:
- It shows how MMU geometry reproduces the correct scaling behaviour
  across the lepton families.
- It does **not** attempt a full field-theoretic Lamb derivation,
  but encodes the empirically validated m^2-scaling as an MMU-effective law.
"""

import math

# ===============================
# 1. Physical constants (SI)
# ===============================

h = 6.62607015e-34        # Planck constant [J s]
hbar = h / (2.0 * math.pi)
c = 299792458.0           # speed of light [m/s]
epsilon0 = 8.8541878128e-12
e_charge = 1.602176634e-19
alpha = 7.2973525693e-3   # fine-structure constant

# Masses [kg]
m_e = 9.1093837015e-31        # electron
m_mu = 1.883531627e-28        # muon
m_tau = 3.16754e-27           # tau
m_p = 1.67262192369e-27       # proton

# g-factors (approximate)
g_e = 2.00231930436256
g_mu = 2.0023318418
g_tau = 2.002  # placeholder

# Conversions
J_to_eV = 1.0 / e_charge


# ===============================
# 2. Reduced mass and Balmer-α
# ===============================

def reduced_mass(m_lepton, m_nucleus):
    """Reduced mass: µ = m_l * m_p / (m_l + m_p)."""
    return (m_lepton * m_nucleus) / (m_lepton + m_nucleus)


def rydberg_constant(mu):
    """
    Rydberg constant for a given reduced mass µ.

    R = µ e^4 / (8 ε0^2 h^3 c)
    (same structure as standard QM, but with variable µ)
    """
    num = mu * e_charge**4
    den = 8.0 * epsilon0**2 * h**3 * c
    return num / den


def balmer_alpha_energy(mu):
    """
    Balmer-α (3 → 2) transition energy for a hydrogenic system:

        E = h c R (1/2^2 - 1/3^2) = h c R * (5/36)
    """
    R = rydberg_constant(mu)
    factor = 5.0 / 36.0
    E_J = h * c * R * factor
    return E_J * J_to_eV  # in eV


def print_balmer_block():
    print("=== Balmer-α (3→2) for electron, muon, tau hydrogen ===")
    for name, m_l in [("electron", m_e), ("muon", m_mu), ("tau", m_tau)]:
        mu = reduced_mass(m_l, m_p)
        E_eV = balmer_alpha_energy(mu)
        # wavelength λ = h c / E
        lam_m = (h * c) / (E_eV * e_charge)
        lam_nm = lam_m * 1e9
        print(f"{name:8s}: E ≈ {E_eV:10.3f} eV,  λ ≈ {lam_nm:8.3f} nm")
    print()


# ===============================
# 3. MMU-effective Lamb model (ΔE ∝ m^2)
# ===============================

"""
MMU-effective Lamb model:

In the MMU hybrid picture,
- w3 is a lepton-dominated torsional mode,
- w4 is a mostly nuclear volumetric mode.

The Lamb shift arises as a very small second-order correction
from the torsion–volume coupling. In the effective description,
this leads to a simple scaling law

    ΔE_Lamb(m_l) = A * m_l^2

where A is a universal MMU parameter.

We determine A from the electronic Lamb shift and then predict
the muonic and tauonic Lamb shifts.

This captures the empirically observed ~m^2 amplification:
    ΔE_mu / ΔE_e ~ (m_mu / m_e)^2
"""

# Experimental Lamb shifts (approximate)
LAMB_E_FREQ_HZ = 1.0578e9          # (2S–2P) in hydrogen
LAMB_E_E_J = h * LAMB_E_FREQ_HZ
LAMB_E_E_eV = LAMB_E_E_J * J_to_eV # ≈ 4.4e-6 eV

LAMB_MU_EXP_eV = 0.20              # muonic hydrogen (2P–2S), rough value


def lamb_prefactor_from_electron():
    """
    Determine A such that ΔE_e = A * m_e^2
    reproduces the observed electron Lamb shift.
    """
    return LAMB_E_E_eV / (m_e**2)


def lamb_shift_MMU_m2(m_l, A):
    """MMU effective prediction: ΔE_l = A * m_l^2 (in eV)."""
    return A * (m_l**2)


def print_lamb_block():
    print("=== MMU effective Lamb model (ΔE ∝ m_l^2) ===")
    print(f"Experimental electron Lamb shift: ~{LAMB_E_E_eV:.3e} eV")
    print(f"Experimental muonic  Lamb shift: ~{LAMB_MU_EXP_eV:.3e} eV\n")

    A = lamb_prefactor_from_electron()
    print(f"Calibrated prefactor A (from electron): {A:.3e} eV / kg^2\n")

    for name, m_l, exp in [
        ("electron", m_e, LAMB_E_E_eV),
        ("muon",    m_mu, LAMB_MU_EXP_eV),
        ("tau",     m_tau, None),
    ]:
        dE = lamb_shift_MMU_m2(m_l, A)
        print(f"{name:8s}: ΔE_Lamb^MMU ≈ {dE:10.3e} eV", end="")
        if exp is not None and name != "electron":
            ratio = dE / exp
            print(f"  (exp ≈ {exp:.3e} eV,  MMU/exp ≈ {ratio:6.2f})")
        else:
            print()

    print()
    # Show the pure m^2 mass ratio
    ratio_m2 = (m_mu / m_e)**2
    print(f"(m_mu/m_e)^2 ≈ {ratio_m2:.3e}  (MMU mass-squared scaling)")
    print(f"Experimental Lamb ratio (µ/e) ≈ {LAMB_MU_EXP_eV / LAMB_E_E_eV:.3e}")
    print()


# ===============================
# 4. Hyperfine (21 cm) scaling
# ===============================

"""
Hyperfine scaling (simple MMU-inspired model):

Hyperfine splitting in the ground state is a spin–spin interaction.
In standard QM:

    ΔE_hfs ∝ μ_l μ_N |ψ(0)|^2,  with  |ψ(0)|^2 ∝ µ^3.

Here we encode this as a mass-based scaling:

    ν_hfs(m_l) = ν_21_H * ( µ_l / µ_e )^3,

where µ_l is the reduced mass for lepton l and a proton nucleus.

This is not a high-precision model for exotic atoms, but it shows
how the MMU torsional sector inherits mass scaling.
"""

NU_21_HZ = 1.420405751e9   # 21 cm line frequency for hydrogen [Hz]


def hyperfine_freq_MMU(m_l):
    """MMU-inspired hyperfine frequency ν_hfs for given lepton mass m_l."""
    mu_e = reduced_mass(m_e, m_p)
    mu_l = reduced_mass(m_l, m_p)
    factor = (mu_l / mu_e)**3
    return NU_21_HZ * factor


def print_hyperfine_block():
    print("=== Hyperfine / 21 cm-like scaling for e-H, µ-H, τ-H ===")
    print(f"Hydrogen 21 cm frequency (exp): {NU_21_HZ/1e6:.6f} MHz\n")
    for name, m_l in [("electron", m_e), ("muon", m_mu), ("tau", m_tau)]:
        nu = hyperfine_freq_MMU(m_l)
        print(f"{name:8s}: ν_hfs^MMU ≈ {nu/1e6:12.3f} MHz")
    print()


# ===============================
# 5. Zeeman / Larmor
# ===============================

"""
Larmor frequency per Tesla:

    ω_L = g q B / (2 m)
    ν_L = ω_L / (2π) = g q / (4π m)   for B = 1 T.

We compute ν_L [Hz/T] for e, µ, τ. The electron case
should reproduce ~28.0 GHz/T.
"""


def larmor_freq_per_T(g, m):
    """Return Larmor frequency ν_L per Tesla [Hz/T]."""
    return g * e_charge / (4.0 * math.pi * m)


def print_zeeman_block():
    print("=== Zeeman / Larmor frequency per Tesla ===")
    for name, g, m in [
        ("electron", g_e, m_e),
        ("muon",    g_mu, m_mu),
        ("tau",     g_tau, m_tau),
    ]:
        nu_per_T = larmor_freq_per_T(g, m)
        print(f"{name:8s}: ν_L ≈ {nu_per_T/1e9:10.3f} GHz/T")
    print()


# ===============================
# 6. Main
# ===============================

if __name__ == "__main__":
    # 1) Balmer (e, µ, τ)
    print_balmer_block()

    # 2) Lamb effective m^2-scaling (e, µ, τ)
    print_lamb_block()

    # 3) Hyperfine / 21 cm-like scaling
    print_hyperfine_block()

    # 4) Zeeman / Larmor frequencies per Tesla
    print_zeeman_block()
