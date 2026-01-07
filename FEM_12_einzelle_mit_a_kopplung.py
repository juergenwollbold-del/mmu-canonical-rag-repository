import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. MMU-kalibrierte Federkonstanten bei a_int = 1 ----------

alpha = 1/137.0          # Feinstruktur
k3_0  = 1.0              # Torsion (Normierung)
k2_0  = 2 * alpha * k3_0 # k2/k3 = 2α
k4_0  = 0.1              # inertial / Masse, hier relativ steif

# Kopplungen klein halten (nur leichte Mischung)
k23_0 = 0.01
k24_0 = 1e-3
k34_0 = 1e-4

K0 = np.array([[k2_0,  k23_0, k24_0],
               [k23_0, k3_0,  k34_0],
               [k24_0, k34_0, k4_0]], dtype=float)

M    = np.eye(3)
Minv = np.linalg.inv(M)

# leichter Dämpfungstensor, damit sich das System einschwingt
C = np.diag([0.01, 0.01, 0.01])

# Geometrie-Kopplung: wie stark w4 die Zellgröße ändert
gamma = 0.2   # gerne spielen!

def build_K(w4):
    """
    a_int = a0 * (1 + gamma*w4), a0=1
    alle Federn ∝ 1/a_int^3
    """
    a_hat = 1.0 + gamma * w4
    a_hat = max(0.5, a_hat)   # Sicherheitsgrenze
    scale = 1.0 / (a_hat**3)
    return K0 * scale, a_hat

# Referenz-Eigenfrequenzen bei a_hat = 1
K_lin, _ = build_K(0.0)
evals, _ = np.linalg.eig(Minv @ K_lin)
evals = np.clip(evals.real, 0.0, None)
omega = np.sqrt(evals); omega.sort()
print("Eigenfrequenzen bei a_hat=1:", omega)

# ---------- 2. Zeitdiskretisierung ------------------------------------

dt   = 0.05 / omega[-1]   # kleiner Zeitschritt
Tend = 500.0
n    = int(Tend / dt)
time = np.arange(n) * dt

# ---------- 3. Anfangsbedingungen -------------------------------------

# Start: elektrische Achse w2 leicht angeregt
w = np.array([1.0, 0.0, 0.0], dtype=float)
v = np.zeros(3, dtype=float)

# Historien
W_hist     = []
T_hist     = []
V_hist     = []
E_hist     = []
a_hat_hist = []
k2_hist, k3_hist, k4_hist = [], [], []

# ---------- 4. Zeitintegration (Velocity-Verlet) ----------------------

F = np.zeros(3)   # noch keine externen Felder E,B,S

for _ in range(n):
    # Schritt 1: aktuelle Steifigkeit aus w4
    K, a_hat = build_K(w[2])
    a = Minv @ (F - C @ v - K @ w)

    v_half = v + 0.5 * dt * a
    w      = w + dt * v_half

    # Schritt 2: neues K, neue Beschleunigung
    K, a_hat = build_K(w[2])
    a_new = Minv @ (F - C @ v_half - K @ w)
    v     = v_half + 0.5 * dt * a_new

    # Energien
    T = 0.5 * v @ (M @ v)
    V = 0.5 * w @ (K @ w)
    E = T + V

    # speichern
    W_hist.append(w.copy())
    T_hist.append(T)
    V_hist.append(V)
    E_hist.append(E)
    a_hat_hist.append(a_hat)
    k2_hist.append(K[0, 0])
    k3_hist.append(K[1, 1])
    k4_hist.append(K[2, 2])

W_hist     = np.array(W_hist)
T_hist     = np.array(T_hist)
V_hist     = np.array(V_hist)
E_hist     = np.array(E_hist)
a_hat_hist = np.array(a_hat_hist)
k2_hist    = np.array(k2_hist)
k3_hist    = np.array(k3_hist)
k4_hist    = np.array(k4_hist)

# ---------- 5. Plots ---------------------------------------------------

# (a) w2,w3,w4
plt.figure()
plt.plot(time, W_hist[:,0], label="w2 (E)")
plt.plot(time, W_hist[:,1], label="w3 (B/Torsion)")
plt.plot(time, W_hist[:,2], label="w4 (Masse/Geom.)")
plt.xlabel("τ")
plt.ylabel("w")
plt.legend()
plt.tight_layout()

# (b) Gesamtenergie
plt.figure()
plt.plot(time, E_hist, label="E = T+V")
plt.xlabel("τ")
plt.ylabel("E")
plt.legend()
plt.tight_layout()

# (c) effektive Federn
plt.figure()
plt.plot(time, k2_hist, label="k2_eff")
plt.plot(time, k3_hist, label="k3_eff")
plt.plot(time, k4_hist, label="k4_eff")
plt.xlabel("τ")
plt.ylabel("k_eff")
plt.legend()
plt.tight_layout()

# (d) innere Länge a_hat
plt.figure()
plt.plot(time, a_hat_hist, label="a_hat")
plt.xlabel("τ")
plt.ylabel("a_int / a0")
plt.legend()
plt.tight_layout()

plt.show()
