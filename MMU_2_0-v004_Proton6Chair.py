import numpy as np
import matplotlib.pyplot as plt

# ============================================
#  Physical constants (SI)
# ============================================
alpha = 1 / 137.035999084      # fine-structure constant
hbar  = 1.054571817e-34        # J·s
c     = 2.99792458e8           # m/s
eV    = 1.602176634e-19        # J
MeV   = 1e6 * eV               # J
m_p   = 1.67262192369e-27      # kg (proton mass)

# Proton radius / chair ring radius (≈ proton charge radius)
a_p = 0.84e-15                 # m

# ============================================
#  MMU spring constants per cell
#  k2 ~ Coulomb spring, k3 ~ torsion spring
# ============================================
hbar_c = hbar * c              # J·m

k2 = 2 * alpha * hbar_c / a_p**3   # J/m^2  (Coulomb / w2)
k3 = 1 * hbar_c / a_p**3           # J/m^2  (torsion / w3)

# Effective ring stiffness (first approximation):
# all 6 sectors effectively feel k2 + k3
k_ring = k2 + k3                  # J/m^2

print("=== Spring constants (MMU, proton chair) ===")
print(f"k2     = {k2:.3e} J/m^2   (Coulomb / w2)")
print(f"k3     = {k3:.3e} J/m^2   (Torsion  / w3)")
print(f"k_ring = {k_ring:.3e} J/m^2   (effective ring stiffness)")

# ============================================
#  Chair ring: 6 masses, 6 springs (periodic)
#  Dimensionless stiffness matrix K for k_ring
# ============================================
K_dimless = np.array([
    [ 2, -1,  0,  0,  0, -1],
    [-1,  2, -1,  0,  0,  0],
    [ 0, -1,  2, -1,  0,  0],
    [ 0,  0, -1,  2, -1,  0],
    [ 0,  0,  0, -1,  2, -1],
    [-1,  0,  0,  0, -1,  2]
], dtype=float)

# Scale with k_ring
K = k_ring * K_dimless          # J/m^2

# ============================================
#  Mass matrix: 6 equal masses on the ring
# ============================================
m_eff = m_p / 6                 # kg (proton mass distributed over 6 sectors)
M = np.eye(6) * m_eff

# ============================================
#  General eigenvalue problem:
#  K v = λ M v  ==>  M^{-1} K v = λ v
#  λ has dimension [1/s^2], ω = sqrt(λ)
# ============================================
Minv = np.linalg.inv(M)
A = Minv @ K

eigvals, eigvecs = np.linalg.eigh(A)   # λ_j

# Set tiny negative eigenvalues to 0 (rounding errors)
eigvals[eigvals < 0] = 0.0

# Eigenfrequencies
omega = np.sqrt(eigvals)              # 1/s

# Energies: E_j = ħ ω_j
E_J   = hbar * omega                  # Joule
E_MeV = E_J / MeV                     # MeV

print("\n=== Eigenvalues and eigenfrequencies (proton chair) ===")
for j, (lam, om, E) in enumerate(zip(eigvals, omega, E_MeV)):
    print(f"Mode {j}: λ = {lam:.3e}  ω = {om:.3e}  E = {E:.3f} MeV")

# Frequency ratios relative to the first nontrivial mode
nonzero_indices = np.where(eigvals > 0)[0]
if len(nonzero_indices) > 0:
    ref = nonzero_indices[0]
    print("\n=== Frequency ratios (normalized to first nontrivial mode) ===")
    for j in nonzero_indices:
        print(f"ω_{j}/ω_{ref} = {omega[j]/omega[ref]:.6f}")
else:
    print("Warning: no nontrivial eigenvalues found.")


# ============================================
#  Geometric hexagon layout for visualization
# ============================================

# 6 points on unit circle (regular hexagon)
angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
x0 = np.cos(angles)
y0 = np.sin(angles)

def plot_mode(mode_index, eigvecs, scale=0.3):
    """
    Plot a single eigenmode as a deformation of the 6 mass points
    on the hexagonal proton chair ring.

    mode_index : index of the mode (0..5)
    eigvecs    : matrix of eigenvectors (columns are modes)
    scale      : graphical amplification factor for displacements
    """
    v = eigvecs[:, mode_index].real

    # Normalize to maximum amplitude ±1
    vmax = np.max(np.abs(v))
    if vmax > 0:
        v = v / vmax

    # Radial displacement along each hexagon radius
    # (project amplitude onto the corresponding radial direction)
    x = x0 + scale * v * x0
    y = y0 + scale * v * y0

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # Lines between neighboring nodes
    for i in range(6):
        j = (i + 1) % 6
        ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', lw=1)

    # Nodes
    ax.scatter(x, y, s=200, c='red', zorder=3)

    # Node indices
    for i in range(6):
        ax.text(x[i]*1.08, y[i]*1.08, f"{i}", fontsize=11,
                ha='center', va='center')

    ax.set_title(f"Proton Chair Mode {mode_index}", fontsize=14)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================
#  Visualization: draw all modes
# ============================================

def plot_all_modes(eigvecs):
    """
    Plot all 6 modes one after another as hexagon deformation shapes.
    """
    for j in range(6):
        plot_mode(j, eigvecs, scale=0.35)


# ============================================
#  Bar plot of eigenenergies in MeV
# ============================================

def plot_energy_spectrum(E_MeV):
    modes = np.arange(len(E_MeV))
    plt.figure(figsize=(6, 4))
    plt.bar(modes, E_MeV, color='C0')
    plt.xlabel("Mode index", fontsize=12)
    plt.ylabel("Energy $E_j$ [MeV]", fontsize=12)
    plt.title("Proton Chair Eigenmode Energies", fontsize=14)
    plt.xticks(modes)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# ============================================
#  Plot of normalized frequency ratios
# ============================================

def plot_frequency_ratios(omega, eigvals):
    nonzero = np.where(eigvals > 0)[0]
    if len(nonzero) == 0:
        print("No nontrivial modes to plot frequency ratios.")
        return

    ref = nonzero[0]
    ratios = omega[nonzero] / omega[ref]

    plt.figure(figsize=(6, 4))
    plt.plot(nonzero, ratios, 'o-', lw=2)
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Mode index", fontsize=12)
    plt.ylabel(r"$\omega_j / \omega_1$", fontsize=12)
    plt.title("Normalized Proton Chair Frequencies", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(nonzero)
    plt.tight_layout()
    plt.show()


# ============================================
#  Main
# ============================================

if __name__ == "__main__":
    # Plot all modes as hexagon mode shapes
    plot_all_modes(eigvecs)

    # Energy spectrum in MeV
    plot_energy_spectrum(E_MeV)

    # Frequency ratios
    plot_frequency_ratios(omega, eigvals)
