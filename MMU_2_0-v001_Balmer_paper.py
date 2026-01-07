import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  MMU prediction: Balmer Hα shift vs gravitational compression
# ============================================================
# Idea:
#   - Internal length a(G) = a0 * (1 - eps_G)
#   - All stiffnesses scale as k_i ~ a^-3
#   - Projected energy difference for a Balmer transition scales as:
#         ΔE(G) = ΔE0 * (1 + 3 * eps_G)   (MMU result)
#   - Wavelength λ = h c / ΔE  =>  λ(G) = λ0 / (1 + 3 * eps_G)
#   - We plot the relative wavelength shift:
#         Δλ/λ = (λ(G) - λ0) / λ0

# -----------------------------
# Parameters
# -----------------------------
lambda0_nm = 656.28  # Hα in vacuum [nm] (reference value)
eps_min = 1e-8
eps_max = 1e-3
N = 300

# Log-spaced eps_G values (gravimetric compression)
eps_G = np.logspace(np.log10(eps_min), np.log10(eps_max), N)

# -----------------------------
# MMU scaling relations
# -----------------------------
# Energy enhancement factor: ΔE(G)/ΔE0
E_factor = 1.0 + 3.0 * eps_G

# Wavelength under MMU gravimetric compression
lambda_G = lambda0_nm / E_factor

# Relative wavelength shift Δλ/λ
delta_lambda_rel = (lambda_G - lambda0_nm) / lambda0_nm

# For reference: absolute shift in pm (picometers)
delta_lambda_pm = (lambda_G - lambda0_nm) * 1e3  # nm -> pm

# -----------------------------
# Plot Δλ/λ vs eps_G
# -----------------------------
fig, ax1 = plt.subplots(figsize=(7, 5))

ax1.set_xscale('log')
ax1.plot(eps_G, delta_lambda_rel, label=r'$\Delta\lambda/\lambda$ (MMU)')

ax1.set_xlabel(r'gravimetric compression $\varepsilon_G$')
ax1.set_ylabel(r'relative wavelength shift $\Delta\lambda/\lambda$')
ax1.grid(True, which='both', linestyle='--', alpha=0.4)

# Optional markers for typical objects (order-of-magnitude)
eps_WD = 1e-6     # white dwarf
eps_NS = 1e-4     # neutron star
eps_MAG = 1e-3    # magnetar

for eps, name in [(eps_WD, 'WD'), (eps_NS, 'NS'), (eps_MAG, 'Magnetar')]:
    # find nearest index
    idx = np.argmin(np.abs(eps_G - eps))
    ax1.scatter(eps_G[idx], delta_lambda_rel[idx], s=30, marker='o')
    ax1.text(eps_G[idx]*1.05, delta_lambda_rel[idx],
             name, fontsize=8, va='bottom')

ax1.set_title('MMU prediction for Hα Balmer shift vs gravimetric compression')

plt.tight_layout()
plt.savefig("201_MMU_Balmer_epsG.png", dpi=300)
plt.show()

# -----------------------------
# Print a few reference values
# -----------------------------
print("Reference Hα wavelength λ0 = {:.5f} nm".format(lambda0_nm))
for eps, name in [(eps_WD, 'White dwarf'),
                  (eps_NS, 'Neutron star'),
                  (eps_MAG, 'Magnetar')]:
    factor = 1.0 + 3.0 * eps
    lam = lambda0_nm / factor
    drel = (lam - lambda0_nm) / lambda0_nm
    dpm = (lam - lambda0_nm) * 1e3
    print(f"{name}: eps_G = {eps:.1e}, Δλ/λ ≈ {drel:.3e}, Δλ ≈ {dpm:.3e} pm")
