import numpy as np, math

# ---------- constants ----------
hbar = 1.054571817e-34
c    = 299792458.0
MeV_to_J = 1.602176634e-13
hc_MeVfm = 197.3269804

# ---------- lepton masses (MeV) ----------
me, mmu, mtau = 0.51099895, 105.6583755, 1776.86
m_lep = np.array([me, mmu, mtau])
s = np.sqrt(m_lep)

# ---------- fit phi, M from m_i = M * A_i^2 with A_i = 1 + sqrt(2) cos(phi+2pi i/3) ----------
phis = np.linspace(0, 2*np.pi, 400000)
best = None
for phi in phis:
    A = np.array([1+np.sqrt(2)*np.cos(phi+2*np.pi*i/3) for i in range(3)])
    # fit sqrt(M): s ≈ sqrt(M)*A
    sqrtM = (s@A)/(A@A)
    err = np.sum((s - sqrtM*A)**2)
    if best is None or err < best[0]:
        best = (err, phi, sqrtM, A)

err, phi, sqrtM, A = best
M = sqrtM**2
print("Lepton fit:")
print("phi =", phi)
print("A   =", A)
print("M(MeV) =", M)
print("m_pred(MeV) =", M*A**2, "  m_exp(MeV) =", m_lep)

# ---------- derive a_eff from M using xi geometry factor ----------
xi = 3.0
# M c^2 = 0.5 * xi * (hbar c / a)  => a = xi * hbar / (2 M c)
M_J = M * MeV_to_J
a_eff = xi * hbar / (2*M_J/c)   # meters
a_eff_fm = a_eff * 1e15
print("\na_eff from torsion+xi:")
print("xi =", xi, "a_eff =", a_eff_fm, "fm")

# ---------- shear modulus from a_eff (MMU torsion scaling) ----------
G = (hbar*c)/(a_eff**4)   # Pa
rho = G/(c**2)
print("\nMaterial params from a_eff:")
print("G =", G, "Pa")
print("rho =", rho, "kg/m^3")

# ---------- isotope check: mu^3 (H,D,T) ----------
m_e = me
m_p, m_d, m_t = 938.27208816, 1875.61294257, 2808.92113298  # MeV
def mu_red(mN): return (m_e*mN)/(m_e+mN)
muH, muD, muT = mu_red(m_p), mu_red(m_d), mu_red(m_t)
print("\nIsotope mu^3 check (relative to H):")
for lab, mu in [("H",muH),("D",muD),("T",muT)]:
    print(lab, "mu/muH=", mu/muH, " (mu/muH)^3=", (mu/muH)**3)

# ---------- 21cm hyperfine isotope check (Fermi scaling) ----------
nu_H = 1420_405_751.768
nu_D = 327_384_352.5222
nu_T = 1_516_701_470.7

mu_p = 2.792_847_344_63
mu_d = 0.857_438_233_5
mu_t = 2.978_962_465_0
I_p, I_d, I_t = 0.5, 1.0, 0.5

def pred_ratio(muI, I, mu_r, muI_ref, I_ref, mu_r_ref):
    fac = (muI/I)*(I+0.5)
    ref = (muI_ref/I_ref)*(I_ref+0.5)
    return (fac/ref) * (mu_r/mu_r_ref)**3

def ppm(pred, meas): return 1e6*(pred/meas - 1.0)

mu_r_H, mu_r_D, mu_r_T = muH, muD, muT
meas_DH, meas_TH = nu_D/nu_H, nu_T/nu_H
pred_DH = pred_ratio(mu_d, I_d, mu_r_D, mu_p, I_p, mu_r_H)
pred_TH = pred_ratio(mu_t, I_t, mu_r_T, mu_p, I_p, mu_r_H)

print("\n21cm isotope check:")
print("D/H meas", meas_DH, "pred", pred_DH, "ppm", ppm(pred_DH, meas_DH))
print("T/H meas", meas_TH, "pred", pred_TH, "ppm", ppm(pred_TH, meas_TH))
# ---------- reporting helpers ----------
def rel_ppm(x_pred, x_meas):
    return 1e6*(x_pred/x_meas - 1.0)

def invert_da_over_a_from_ppm(ppm_err, power=-3):
    # If nu ∝ a^(power), then dnu/nu = power * da/a  => da/a = (dnu/nu)/power
    dnu_over_nu = ppm_err * 1e-6
    return dnu_over_nu / power

print("\n=== Lepton fit errors ===")
m_pred = M*A**2
for name, mp, mx in zip(["e","mu","tau"], m_pred, m_lep):
    print(f"{name:>3}: pred={mp:12.8f}  exp={mx:12.8f}  ppm={rel_ppm(mp,mx):+.3f}")

print("\n=== 21cm isotope errors + inferred da/a (nu ∝ a^-3) ===")
for lab, pred, meas in [("D/H", pred_DH, meas_DH), ("T/H", pred_TH, meas_TH)]:
    ppm_err = rel_ppm(pred, meas)
    da_over_a = invert_da_over_a_from_ppm(ppm_err, power=-3)
    print(f"{lab}: meas={meas:.15f}  pred={pred:.15f}  ppm={ppm_err:+.3f}  da/a={da_over_a:+.6e}")

# optional: interpret as effective torsion-coupling correction k34 (same scaling as a^-3)
print("\n=== Equivalent delta(k_eff)/k_eff assuming nu ∝ k_eff ===")
for lab, pred, meas in [("D/H", pred_DH, meas_DH), ("T/H", pred_TH, meas_TH)]:
    ppm_err = rel_ppm(pred, meas)
    print(f"{lab}: dk/k ≈ dnu/nu = {ppm_err*1e-6:+.6e}")
