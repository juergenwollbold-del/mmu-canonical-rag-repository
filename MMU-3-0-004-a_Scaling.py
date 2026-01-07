import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# MMU-Isotopen-Maschine
# -----------------------------------------

m_e = 1.0
m_p = 1836.152

def mu_reduced(mN):
    return (m_e * mN) / (m_e + mN)

def mmu_observables_from_r(r):
    return {
        "Balmer": r**(-1),
        "Zeeman": r**(+1),
        "Lamb":   r**(-3),
    }

systems = [
    ("H",  1, [1, 2, 3]),
    ("He", 2, [3, 4]),
    ("Li", 3, [6, 7]),
    ("O",  8, [16, 18]),
    ("Ca", 20, [40, 44])
]

rows = []

for elem, Z, A_list in systems:
    A_ref = min(A_list)
    mu_ref = mu_reduced(A_ref * m_p)
    
    for A in A_list:
        mu_iso = mu_reduced(A * m_p)
        r = mu_ref / mu_iso
        obs = mmu_observables_from_r(r)
        
        if A == A_ref:
            dBalmer = 0.0
            dZeeman = 0.0
            dLamb   = 0.0
        else:
            dBalmer = obs["Balmer"] - 1.0
            dZeeman = obs["Zeeman"] - 1.0
            dLamb   = obs["Lamb"]   - 1.0
        
        rows.append({
            "Element": elem,
            "A": A,
            "r": r,
            "Balmer_shift": dBalmer,
            "Zeeman_shift": dZeeman,
            "Lamb_shift": dLamb
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# -----------------------------------------
# Plot: Shifts for all isotopes
# -----------------------------------------
plt.figure(figsize=(8, 6))
x = np.arange(len(df))

plt.plot(x, df["Balmer_shift"], label="Balmer shift")
plt.plot(x, df["Zeeman_shift"], label="Zeeman shift")
plt.plot(x, df["Lamb_shift"], label="Lamb shift")

plt.xticks(x, df["Element"] + "-" + df["A"].astype(str), rotation=45)
plt.ylabel("Relative shift")
plt.title("MMU Predicted Isotope Shifts (Balmer, Zeeman, Lamb)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# -----------------------------------------
# Plot: Shifts as function of a/a0
# -----------------------------------------

r = np.linspace(0.98, 1.02, 400)  # small span around 1
balmer = r**(-1) - 1
zeeman = r - 1
lamb = r**(-3) - 1

plt.figure(figsize=(8,6))
plt.plot(r, balmer, label="Balmer shift")
plt.plot(r, zeeman, label="Zeeman shift")
plt.plot(r, lamb, label="Lamb shift")
plt.xlabel("r = a/a0")
plt.ylabel("Relative shift")
plt.title("MMU Predicted Shifts as Function of a/a0")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
