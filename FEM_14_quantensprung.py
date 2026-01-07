import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. MMU-Federkonstanten bei a_int = 1 -----------------------

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
C    = np.diag([0.005, 0.005, 0.005])

gamma = 0.2              # Geometrie-Kopplung

def K_from_a_hat(a_hat):
    """Skalierung aller Federn mit a_hat."""
    scale = 1.0 / (a_hat**3)
    return K0 * scale

# Referenz-Eigenfrequenzen
K_lin = K_from_a_hat(1.0)
evals, _ = np.linalg.eig(Minv @ K_lin)
evals = np.clip(evals.real, 0.0, None)
omega = np.sqrt(evals); omega.sort()
print("Eigenfrequenzen bei a_hat=1:", omega)

# Zeitdiskretisierung (für alle Runs gleich)
dt   = 0.05 / omega[-1]
Tend_sweep = 300.0
Tend_jump  = 500.0

# E-Feld-Parameter
chi2 = 1.0
E0   = 0.1

# ---------- 2. Hilfsfunktion: eine Simulation --------------------------

def simulate_drive(omega_drive, Tend, quantum_jump=False,
                   a_levels=(1.0, 1.1), a_crit=1.05):
    """
    omega_drive: Anregungsfrequenz des E-Felds
    quantum_jump=False: kontinuierliche Geometrie
    quantum_jump=True: a_hat springt diskret von a0->a1, wenn a_cont > a_crit
    """
    n = int(Tend / dt)
    time = np.arange(n) * dt

    w = np.zeros(3, dtype=float)
    v = np.zeros(3, dtype=float)

    level = 0                   # 0 -> Grundzustand (a=1.0), 1 -> angeregter Zustand (a=1.1)
    jump_time = None

    W_hist = []
    a_hist = []

    for i in range(n):
        t = time[i]
        E_t = E0 * np.sin(omega_drive * t)
        F   = np.array([chi2 * E_t, 0.0, 0.0])

        # kontinuierliche Geometrie aus w4
        a_cont = 1.0 + gamma * w[2]
        a_cont = max(0.5, a_cont)

        if quantum_jump:
            # Wenn noch im Grundzustand und Geometrie zu groß -> Quantensprung
            if level == 0 and a_cont >= a_crit:
                level = 1
                jump_time = t
            # diskretes a_hat
            a_hat = a_levels[level]
        else:
            a_hat = a_cont

        K = K_from_a_hat(a_hat)
        a = Minv @ (F - C @ v - K @ w)

        v_half = v + 0.5 * dt * a
        w      = w + dt * v_half

        # zweiter Halbschritt mit Feld bei t+dt
        t_new = t + dt
        E_new = E0 * np.sin(omega_drive * t_new)
        F_new = np.array([chi2 * E_new, 0.0, 0.0])

        # Geometrie erneut bestimmen (für Konsistenz: gleiche Logik)
        a_cont_new = 1.0 + gamma * w[2]
        a_cont_new = max(0.5, a_cont_new)
        if quantum_jump:
            if level == 0 and a_cont_new >= a_crit:
                level = 1
                jump_time = t_new
            a_hat_new = a_levels[level]
        else:
            a_hat_new = a_cont_new

        K_new = K_from_a_hat(a_hat_new)
        a_new = Minv @ (F_new - C @ v_half - K_new @ w)
        v     = v_half + 0.5 * dt * a_new

        W_hist.append(w.copy())
        a_hist.append(a_hat_new)

    W_hist = np.array(W_hist)
    a_hist = np.array(a_hist)
    max_w2 = np.max(np.abs(W_hist[:, 0]))
    return time, W_hist, a_hist, max_w2, jump_time

# ---------- 3. Resonanz-Sweep ohne Quantensprung -----------------------

omega2 = omega[1]  # "elektrischer" Mode
omega_drives = np.linspace(0.5*omega2, 1.5*omega2, 25)

amp_list = []
for od in omega_drives:
    _, W_tmp, _, max_w2, _ = simulate_drive(od, Tend_sweep, quantum_jump=False)
    amp_list.append(max_w2)

amp_list = np.array(amp_list)

# ---------- 4. Ein Lauf mit Quantensprung-Logik -----------------------

time_q, W_q, a_q, _, jump_time = simulate_drive(omega2, Tend_jump,
                                                quantum_jump=True,
                                                a_levels=(1.0, 1.1),
                                                a_crit=1.05)

print("Quantensprung-Zeit (falls None -> kein Sprung):", jump_time)

# ---------- 5. Plots ---------------------------------------------------

# (A) Resonanzkurve: max |w2| vs omega_drive
plt.figure()
plt.plot(omega_drives, amp_list, 'o-')
plt.xlabel(r"$\omega_\mathrm{drive}$")
plt.ylabel(r"$\max|w_2|$")
plt.title("Resonanzkurve (ohne Quantensprung)")
plt.tight_layout()

# (B) Zeitverlauf w2,w3,w4 mit Quantensprung
plt.figure()
plt.plot(time_q, W_q[:,0], label="w2 (E)")
plt.plot(time_q, W_q[:,1], label="w3 (Torsion)")
plt.plot(time_q, W_q[:,2], label="w4 (Geom.)")
if jump_time is not None:
    plt.axvline(jump_time, color='k', linestyle='--', label="Quantensprung")
plt.xlabel("τ")
plt.ylabel("w")
plt.legend()
plt.title("Zeitverlauf mit Quantensprung-Mechanik")
plt.tight_layout()

# (C) a_hat(t) mit Sprung
plt.figure()
plt.plot(time_q, a_q, label="a_hat")
if jump_time is not None:
    plt.axvline(jump_time, color='k', linestyle='--', label="Quantensprung")
plt.xlabel("τ")
plt.ylabel("a_int / a0")
plt.legend()
plt.title("Geometrie a_hat(t)")
plt.tight_layout()

plt.show()
