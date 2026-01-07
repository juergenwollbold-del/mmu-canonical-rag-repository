import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. MMU-kalibrierte Federkonstanten bei a_int = 1 ----------

alpha = 1/137.0
k3_0  = 1.0
k2_0  = 2 * alpha * k3_0
k4_0  = 0.1

k23_0 = 0.01
k24_0 = 1e-3
k34_0 = 1e-4

K0 = np.array([[k2_0,  k23_0, k24_0],
               [k23_0, k3_0,  k34_0],
               [k24_0, k34_0, k4_0]], dtype=float)

M    = np.eye(3)
Minv = np.linalg.inv(M)

C = np.diag([0.005, 0.005, 0.005])   # leichte Dämpfung

gamma = 0.2   # Geometrie-Kopplung w4 -> a_hat

def build_K(w4):
    a_hat = 1.0 + gamma * w4
    a_hat = max(0.5, a_hat)
    scale = 1.0 / (a_hat**3)
    return K0 * scale, a_hat

# Referenz-Eigenfrequenzen
K_lin, _ = build_K(0.0)
evals, _ = np.linalg.eig(Minv @ K_lin)
evals = np.clip(evals.real, 0.0, None)
omega = np.sqrt(evals); omega.sort()
print("Eigenfrequenzen bei a_hat=1:", omega)

# Anregung: externes elektrisches Feld E(t)
chi2 = 1.0         # Kopplungsfaktor Feld -> w2
E0   = 0.1         # Feldamplitude
omega_drive = omega[1]   # in Nähe des 2. Modes

def E_of_t(t):
    return E0 * np.sin(omega_drive * t)

# ---------- 2. Zeitdiskretisierung ------------------------------------

dt   = 0.05 / omega[-1]
Tend = 500.0
n    = int(Tend / dt)
time = np.arange(n) * dt

# ---------- 3. Anfangsbedingungen -------------------------------------

w = np.array([0.0, 0.0, 0.0], dtype=float)  # Start in Ruhe
v = np.zeros(3, dtype=float)

W_hist     = []
E_hist     = []
a_hat_hist = []

# ---------- 4. Zeitintegration mit E(t) -------------------------------

for i in range(n):
    t      = time[i]
    E_t    = E_of_t(t)
    F      = np.array([chi2 * E_t, 0.0, 0.0])

    K, a_hat = build_K(w[2])
    a = Minv @ (F - C @ v - K @ w)

    v_half = v + 0.5 * dt * a
    w      = w + dt * v_half

    t_new  = t + dt
    E_new  = E_of_t(t_new)
    F_new  = np.array([chi2 * E_new, 0.0, 0.0])

    K, a_hat = build_K(w[2])
    a_new = Minv @ (F_new - C @ v_half - K @ w)
    v     = v_half + 0.5 * dt * a_new

    # Gesamtenergie (ohne explizites Feldenergie-Term)
    T = 0.5 * v @ (M @ v)
    V = 0.5 * w @ (K @ w)
    E_tot = T + V

    W_hist.append(w.copy())
    E_hist.append(E_tot)
    a_hat_hist.append(a_hat)

W_hist     = np.array(W_hist)
E_hist     = np.array(E_hist)
a_hat_hist = np.array(a_hat_hist)

# ---------- 5. Plots ---------------------------------------------------

plt.figure()
plt.plot(time, W_hist[:,0], label="w2 (E)")
plt.plot(time, W_hist[:,1], label="w3 (B/Torsion)")
plt.plot(time, W_hist[:,2], label="w4 (Masse/Geom.)")
plt.xlabel("τ"); plt.ylabel("w"); plt.legend(); plt.tight_layout()

plt.figure()
plt.plot(time, E_hist, label="E_total")
plt.xlabel("τ"); plt.ylabel("E"); plt.legend(); plt.tight_layout()

plt.figure()
plt.plot(time, a_hat_hist, label="a_hat")
plt.xlabel("τ"); plt.ylabel("a_int/a0"); plt.legend(); plt.tight_layout()

plt.show()
