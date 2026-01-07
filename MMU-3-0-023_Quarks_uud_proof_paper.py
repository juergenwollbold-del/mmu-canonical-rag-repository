import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# 1. Physikalische Konstanten und MMU-Steifigkeiten k2(a), k3(a), k4(a)
# ================================================================

alpha = 1 / 137.035999084      # Feinstrukturkonstante
hbar  = 1.054571817e-34        # J·s
c     = 2.99792458e8           # m/s
eV    = 1.602176634e-19        # J
MeV   = 1e6 * eV               # J
m_p   = 1.67262192369e-27      # kg (Protonmasse)
G     = 6.67430e-11            # m^3 kg^-1 s^-2

hbar_c = hbar * c              # J·m

def k2_MMU(a):
    """MMU electric stiffness k2(a) = 2 alpha hbar c / a^3"""
    return 2.0 * alpha * hbar_c / a**3

def k3_MMU(a):
    """MMU torsional stiffness k3(a) = hbar c / a^3"""
    return 1.0 * hbar_c / a**3

def k4_MMU(a, m):
    """MMU volumetric stiffness k4(a) = 2 G m^2 / a^3"""
    return 2.0 * G * m**2 / a**3

# Proton-"chair"-Radius
a_p = 0.84e-15  # m

# Proton-Steifigkeiten direkt aus MMU-Formeln
k2_p = k2_MMU(a_p)
k3_p = k3_MMU(a_p)
k_ring_p = k2_p + k3_p

print("=== Proton chair MMU stiffnesses (from MMU k2,k3 formulas) ===")
print(f"k2_p      = {k2_p:.3e} J/m^2   (Coulomb / w2)")
print(f"k3_p      = {k3_p:.3e} J/m^2   (torsion  / w3)")
print(f"k_ring_p  = {k_ring_p:.3e} J/m^2   (effective ring stiffness)")
print()

# Massen (Basiswerte, m_spin_chair wird unten skaliert)
m_eff_chair_base = m_p / 6.0        # radiale Chair-Segmentmasse
m_spin_chair_base = m_p / 2.0       # effektive Chair-Spin-Masse

# ================================================================
# 2. DOF-Zählung
# ================================================================

N_CHAIR_RAD = 6    # 6 radiale Chair-DOFs
N_CHAIR_SPIN = 1   # 1 globaler Chair-w3-DOF
N_CHAIR = N_CHAIR_RAD + N_CHAIR_SPIN  # = 7
N_QUARK = 9        # 3 Quarks x (w2,w3,w4)
DIM = N_CHAIR + N_QUARK  # = 16

# Indizes
IDX_CHAIR_RAD = np.arange(0, N_CHAIR_RAD)  # 0..5
IDX_CHAIR_SPIN = N_CHAIR_RAD              # 6
FIRST_QUARK = N_CHAIR                      # 7
IDX_QUARK = np.arange(FIRST_QUARK, FIRST_QUARK + N_QUARK)

# ================================================================
# 3. Globale Cross-Federn (werden im Scan skaliert)
# ================================================================

BETA23 = 0.2
BETA24 = 0.15
BETA34 = 0.1

# ================================================================
# 4. Quark-Ladungen aus Ring-Eigenvektor (1,1,-2)
# ================================================================

def compute_quark_charges():
    v = np.array([1.0, 1.0, -2.0])
    q0 = 1.0 / 3.0
    alpha_q = 1.0 / 3.0
    return q0 + alpha_q * v   # [2/3, 2/3, -1/3]

# ================================================================
# 5. Lokale 3x3-MMU-Steifigkeit (w2,w3,w4) mit globalen Betas
# ================================================================

def local_K(k2, k3, k4):
    """Use global BETA23/24/34 so we can vary them in scans."""
    global BETA23, BETA24, BETA34
    k23 = BETA23 * np.sqrt(k2 * k3)
    k24 = BETA24 * np.sqrt(k2 * k4)
    k34 = BETA34 * np.sqrt(k3 * k4)
    return np.array([
        [k2,  k23, k24],
        [k23, k3,  k34],
        [k24, k34, k4 ],
    ], dtype=float)

def ring_matrix_axis(k):
    return k * np.array([
        [ 2., -1., -1.],
        [-1.,  2., -1.],
        [-1., -1.,  2.],
    ], dtype=float)

# ================================================================
# 6. Aufbau von K und M für das Gesamtsystem (16x16)
# ================================================================

def build_K_M(epsilon_rad=0.05, epsilon_spin=0.05, m_spin_chair_factor=1.0):
    global m_eff_chair_base, m_spin_chair_base

    K_tot = np.zeros((DIM, DIM), dtype=float)
    M_tot = np.zeros((DIM, DIM), dtype=float)

    # aktuelle Massen
    m_eff_chair = m_eff_chair_base
    m_spin_chair = m_spin_chair_base * m_spin_chair_factor

    # ---------------- Chair-Ring: 6 radiale DOFs (0..5) ----------------
    K_dimless_chair = np.array([
        [ 2, -1,  0,  0,  0, -1],
        [-1,  2, -1,  0,  0,  0],
        [ 0, -1,  2, -1,  0,  0],
        [ 0,  0, -1,  2, -1,  0],
        [ 0,  0,  0, -1,  2, -1],
        [-1,  0,  0,  0, -1,  2]
    ], dtype=float)

    K_chair_rad = k_ring_p * K_dimless_chair
    M_chair_rad = np.eye(N_CHAIR_RAD) * m_eff_chair

    K_tot[np.ix_(IDX_CHAIR_RAD, IDX_CHAIR_RAD)] = K_chair_rad
    M_tot[np.ix_(IDX_CHAIR_RAD, IDX_CHAIR_RAD)] = M_chair_rad

    # ---------------- Chair-Spin-DOF (globaler w3) bei Index 6 ----------------
    i_spin = IDX_CHAIR_SPIN
    # Torsions-Steifigkeit des Rings ~ 6 * k3_p
    K_tot[i_spin, i_spin] = 6.0 * k3_p
    M_tot[i_spin, i_spin] = m_spin_chair

    # ---------------- Quark-Block (9 DOFs) ab Index FIRST_QUARK ----------------
    a_q = a_p / 2.0          # halbe Länge
    m_q = m_p / 3.0          # grob 3 Quarks im Proton

    k2_q = k2_MMU(a_q)
    k3_q = k3_MMU(a_q)
    k4_q = k4_MMU(a_q, m_q)

    K_loc_q = local_K(k2_q, k3_q, k4_q)

    for q in range(3):
        base = FIRST_QUARK + 3*q
        K_tot[base:base+3, base:base+3] += K_loc_q

    R2_q = ring_matrix_axis(k2_q * 0.5)
    R3_q = ring_matrix_axis(k3_q * 0.5)

    for a in range(3):
        for b in range(3):
            ci = FIRST_QUARK + 3*a
            cj = FIRST_QUARK + 3*b
            # w2
            K_tot[ci + 0, cj + 0] += R2_q[a, b]
            # w3
            K_tot[ci + 1, cj + 1] += R3_q[a, b]

    # Quark-Massen
    m_quark_total = m_p / 3.0
    m_quark_dof = m_quark_total / 3.0
    for q in range(3):
        base = FIRST_QUARK + 3*q
        for comp in range(3):
            M_tot[base+comp, base+comp] = m_quark_dof

    # ---------------- Chair–Quark-Kopplung (radial) ----------------
    gamma2_base = 0.1 * k2_p
    gamma3_base = 0.1 * k3_p
    gamma4_base = 0.1 * k3_p

    gamma2 = epsilon_rad * gamma2_base
    gamma3 = epsilon_rad * gamma3_base
    gamma4 = epsilon_rad * gamma4_base

    chair_nodes = [0, 2, 4]  # radiale Chair-DOFs
    for q, node in enumerate(chair_nodes):
        base_q = FIRST_QUARK + 3*q
        for comp, g in enumerate([gamma2, gamma3, gamma4]):
            i = node
            j = base_q + comp
            K_tot[i, j] += g
            K_tot[j, i] += g

    # ---------------- Chair-Spin–Quark-w3-Kopplung ----------------
    gamma3_spin_base = 0.1 * k3_p
    gamma3_spin = epsilon_spin * gamma3_spin_base

    for q in range(3):
        base_q = FIRST_QUARK + 3*q
        j_w3 = base_q + 1
        K_tot[i_spin, j_w3] += gamma3_spin
        K_tot[j_w3, i_spin] += gamma3_spin

    return K_tot, M_tot, a_q

# ================================================================
# 7. Eigenproblem und Energien
# ================================================================

def solve_modes(K, M):
    m_vec = np.diag(M)
    if np.any(m_vec <= 0):
        raise ValueError("Non-positive masses in diagonal of M.")
    Minv_sqrt = np.diag(1.0 / np.sqrt(m_vec))

    K_eff = Minv_sqrt @ K @ Minv_sqrt
    evals, U = np.linalg.eigh(K_eff)
    evals[evals < 0] = 0.0
    omega = np.sqrt(evals)

    idx = np.argsort(omega)
    omega = omega[idx]
    U = U[:, idx]

    V = Minv_sqrt @ U
    E_J = hbar * omega
    E_MeV = E_J / MeV
    return E_MeV, V

# ================================================================
# 8. Effektiver Protonen-g-Faktor
# ================================================================

def compute_proton_g_eff(E_MeV, V, q_charges, a_p, a_q, energy_power=1.0):
    """
    Effektiver g_p:
      - nutzt Quark-w3-Spinverteilung (Ladungen)
      - Chair-Spin-DOF verstärkt das Moment über a_p
      - Gewicht:
          weight_m ~ (Q*w3_quark + C_spin) * a_eff / E^energy_power
    """
    chair_rad_idx = IDX_CHAIR_RAD
    i_spin = IDX_CHAIR_SPIN
    quark_idx = IDX_QUARK

    num = 0.0
    den = 0.0

    for Em, vec in zip(E_MeV, V.T):
        v = vec / np.linalg.norm(vec)

        C_rad = np.sum(v[chair_rad_idx]**2)
        C_spin = v[i_spin]**2
        Q = np.sum(v[quark_idx]**2)

        # Quark w2/w3/w4 + w3-Komponenten je Quark
        w2_q = w3_q = w4_q = 0.0
        q_w3 = []
        for q in range(3):
            base = FIRST_QUARK + 3*q
            w2_q += v[base+0]**2
            w3_q += v[base+1]**2
            w4_q += v[base+2]**2
            q_w3.append(v[base+1])
        q_w3 = np.array(q_w3)

        if (w3_q + C_spin) <= 0 or Em <= 0:
            continue

        if np.sum(q_w3**2) == 0:
            continue
        S = q_w3**2 / np.sum(q_w3**2)
        g_mode = np.sum(q_charges * S)

        # effektiver Hebelarm
        a_eff = (w3_q * a_q + C_spin * a_p) / (w3_q + C_spin)

        # Gewichtung: Quark-Spin + Chair-Spin, skaliert mit Radius / Energie^p
        weight = (Q * w3_q + C_spin) * (a_eff / (Em**energy_power))

        num += weight * g_mode
        den += weight

    q_spin_eff = num / den if den > 0 else 0.0
    g_p_eff = 2.0 * q_spin_eff

    return g_p_eff

# ================================================================
# 9. Hilfsfunktion: ein Modelllauf mit Parametern
# ================================================================

def run_model(eps_rad, eps_spin, beta_scale, m_spin_factor):
    global BETA23, BETA24, BETA34

    # Betas skalieren
    BETA23 = 0.2 * beta_scale
    BETA24 = 0.15 * beta_scale
    BETA34 = 0.1 * beta_scale

    q_charges = compute_quark_charges()
    K, M, a_q = build_K_M(epsilon_rad=eps_rad,
                          epsilon_spin=eps_spin,
                          m_spin_chair_factor=m_spin_factor)
    E_MeV, V = solve_modes(K, M)

    g_p_eff = compute_proton_g_eff(E_MeV, V, q_charges,
                                   a_p=a_p, a_q=a_q,
                                   energy_power=1.0)
    scale = np.sqrt(k3_p / k2_p)
    g_p_phys = scale * g_p_eff
    return g_p_eff, g_p_phys

# ================================================================
# 10. Optional: einfache Spektrums-Analyse für einen Referenzlauf
# ================================================================

def analyze_modes(E, V):
    n_modes = len(E)
    chair_idx = np.arange(0, N_CHAIR)
    quark_idx = IDX_QUARK

    chair_frac = np.zeros(n_modes)
    quark_frac = np.zeros(n_modes)
    w2_frac = np.zeros(n_modes)
    w3_frac = np.zeros(n_modes)
    w4_frac = np.zeros(n_modes)

    print("\nMode analysis (chair/quark and w2/w3/w4 fractions):\n")
    for m, (Em, vec) in enumerate(zip(E, V.T), start=1):
        v = vec / np.linalg.norm(vec)

        chair_amp = np.sum(v[chair_idx]**2)
        quark_amp = np.sum(v[quark_idx]**2)

        w2 = w3 = w4 = 0.0
        for q in range(3):
            base = FIRST_QUARK + 3*q
            w2 += v[base + 0]**2
            w3 += v[base + 1]**2
            w4 += v[base + 2]**2

        chair_frac[m-1] = chair_amp
        quark_frac[m-1] = quark_amp
        w2_frac[m-1] = w2
        w3_frac[m-1] = w3
        w4_frac[m-1] = w4

        print(f"Mode {m:2d}: E = {Em:8.3f} MeV")
        print(f"  chair : {chair_amp:5.3f},  quark : {quark_amp:5.3f}")
        print(f"  w2 fraction: {w2:5.3f},  w3 fraction: {w3:5.3f},  w4 fraction: {w4:5.3f}\n")

    return chair_frac, quark_frac, w2_frac, w3_frac, w4_frac

# ================================================================
# 11. Main mit Robustheits-Scan
# ================================================================

def main():
    # 1) Referenzlauf mit Ausgabe der Eigenenergien + g_p
    print("=== Reference run (detail) ===")
    q_charges = compute_quark_charges()
    print("Quark charges from ring mode (1,1,-2):", q_charges,
          "   sum =", np.sum(q_charges))

    # Referenz-Parameter
    eps_rad_ref = 0.05
    eps_spin_ref = 0.05
    beta_scale_ref = 1.0
    m_spin_factor_ref = 1.0

    BETA23 = 0.2 * beta_scale_ref
    BETA24 = 0.15 * beta_scale_ref
    BETA34 = 0.1 * beta_scale_ref

    K_ref, M_ref, a_q_ref = build_K_M(epsilon_rad=eps_rad_ref,
                                      epsilon_spin=eps_spin_ref,
                                      m_spin_chair_factor=m_spin_factor_ref)
    E_ref, V_ref = solve_modes(K_ref, M_ref)

    print("\nEigenenergies (MMU k2,k3,k4, chair+spin+quark):")
    for i, Em in enumerate(E_ref, start=1):
        print(f"Mode {i:2d}: {Em:8.3f} MeV")

    analyze_modes(E_ref, V_ref)

    g_p_eff_ref = compute_proton_g_eff(E_ref, V_ref, q_charges,
                                       a_p=a_p, a_q=a_q_ref,
                                       energy_power=1.0)
    scale = np.sqrt(k3_p / k2_p)
    g_p_phys_ref = scale * g_p_eff_ref
    print(f"\nReference: g_p_eff ≈ {g_p_eff_ref:.3f}")
    print(f"Stiffness scale factor sqrt(k3_p/k2_p) = {scale:.3f}")
    print(f"Scaled physical g_p_phys ≈ {g_p_phys_ref:.3f}\n")

    # 2) Grober Scan über Parameter
    print("=== Robustness scan over eps_rad, eps_spin, beta_scale, m_spin_factor ===")
    for eps_rad in [0.01, 0.05, 0.1]:
        for eps_spin in [0.01, 0.05, 0.1]:
            for beta_scale in [0.5, 1.0, 2.0]:
                for m_spin_factor in [0.5, 1.0, 2.0]:
                    g_eff, g_phys = run_model(eps_rad, eps_spin,
                                              beta_scale, m_spin_factor)
                    print(f"eps_rad={eps_rad:.2f}, eps_spin={eps_spin:.2f}, "
                          f"beta_scale={beta_scale:.1f}, m_spin_factor={m_spin_factor:.1f} "
                          f"-> g_eff={g_eff:.3f}, g_phys={g_phys:.3f}")

if __name__ == "__main__":
    main()
