import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. MMU-Federkonstanten bei a_int = 1 -----------------------

alpha = 1/137.0
k3_0  = 1.0
k2_0  = 2 * alpha * k3_0
k4_0  = 0.1

k23_0 = 0.05
k24_0 = 0.01
k34_0 = 0.002

K0_base = np.array([[k2_0,  k23_0, k24_0],
                    [k23_0, k3_0,  k34_0],
                    [k24_0, k34_0, k4_0]], dtype=float)

M    = np.eye(3)
Minv = np.linalg.inv(M)
C    = np.diag([0.005, 0.005, 0.005])

gamma = 0.3                # Kopplung w4 -> a_hat

def K_from_a_hat(a_hat):
    scale = 1.0 / (a_hat**3)
    return K0_base * scale

# Eigenfrequenzen bei a_hat = 1 und a_hat = 1.3 (für Vergleich)
K_g = K_from_a_hat(1.0)
evals_g, _ = np.linalg.eig(Minv @ K_g)
omega_g = np.sqrt(np.clip(evals_g.real, 0.0, None)); omega_g.sort()

K_e = K_from_a_hat(1.3)
evals_e, _ = np.linalg.eig(Minv @ K_e)
omega_e = np.sqrt(np.clip(evals_e.real, 0.0, None)); omega_e.sort()

print("ω (a=1.0):", omega_g)
print("ω (a=1.3):", omega_e)
omega2 = omega_g[1]    # „elektrischer“ Mode

# Zeitdiskretisierung
dt   = 0.05 / omega_g[-1]
Tend = 600.0
n    = int(Tend / dt)
time = np.arange(n) * dt

# E-Feld-Parameter
chi2 = 1.0
E0   = 0.2
omega_drive = omega2

# ---------- 2. Simulation: Feld AN vor Sprung, AUS nach Sprung ----------

a_levels = (1.0, 1.3)
a_crit   = 1.05

w = np.zeros(3, dtype=float)
v = np.zeros(3, dtype=float)

level      = 0
field_on   = True
jump_time  = None

W_hist = []
a_hist = []

for i in range(n):
    t = time[i]

    # Feld nur solange level==0
    if field_on:
        E_t = E0 * np.sin(omega_drive * t)
    else:
        E_t = 0.0
    F = np.array([chi2 * E_t, 0.0, 0.0])

    a_cont = 1.0 + gamma * w[2]
    a_cont = max(0.5, a_cont)

    if level == 0 and a_cont >= a_crit:
        level = 1
        field_on = False     # Feld AUS nach Sprung
        jump_time = t

    a_hat = a_levels[level]
    K = K_from_a_hat(a_hat)
    a = Minv @ (F - C @ v - K @ w)

    v_half = v + 0.5 * dt * a
    w      = w + dt * v_half

    # zweiter Halbschritt
    t_new = t + dt
    if field_on:
        E_new = E0 * np.sin(omega_drive * t_new)
    else:
        E_new = 0.0
    F_new = np.array([chi2 * E_new, 0.0, 0.0])

    a_cont_new = 1.0 + gamma * w[2]
    a_cont_new = max(0.5, a_cont_new)
    if level == 0 and a_cont_new >= a_crit:
        level = 1
        field_on = False
        jump_time = t_new

    a_hat_new = a_levels[level]
    K_new = K_from_a_hat(a_hat_new)
    a_new = Minv @ (F_new - C @ v_half - K_new @ w)
    v     = v_half + 0.5 * dt * a_new

    W_hist.append(w.copy())
    a_hist.append(a_hat_new)

W_hist = np.array(W_hist)
a_hist = np.array(a_hist)

print("Quantensprung bei τ ≈", jump_time)

# ---------- 3. Plots ----------------------------------------------------

plt.figure()
plt.plot(time, W_hist[:,0], label="w2 (E)")
plt.plot(time, W_hist[:,1], label="w3 (Torsion)")
plt.plot(time, W_hist[:,2], label="w4 (Geom.)")
if jump_time is not None:
    plt.axvline(jump_time, color='k', linestyle='--', label="Sprung")
plt.xlabel("τ"); plt.ylabel("w"); plt.legend(); plt.tight_layout()

plt.figure()
plt.plot(time, a_hist, label="a_hat")
if jump_time is not None:
    plt.axvline(jump_time, color='k', linestyle='--', label="Sprung")
plt.xlabel("τ"); plt.ylabel("a_int/a0"); plt.legend(); plt.tight_layout()

plt.show()
