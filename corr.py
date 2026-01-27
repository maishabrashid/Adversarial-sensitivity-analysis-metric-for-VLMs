import numpy as np
import matplotlib.pyplot as plt

vulnerability_scores = [2.52, 1.2, 0.89, 0.81, 0.74, 0.67, 0.63, 0.53]
sensitivity_scores = [0.54, 0.55, 0.54, 0.52, 0.51, 0.5, 0.5, 0.51]

#vulnerability_scores = [0.88, 0.23, 0.18, 0.179, 0.171, 0.17, 0.16]
#sensitivity_scores = [42.12, 41.76, 42.81, 42.45, 40.88, 38.59, 37.45]

S  = np.array(sensitivity_scores, dtype=np.float64)
VS = np.array(vulnerability_scores, dtype=np.float64)

from math import sqrt
from scipy.special import betainc  # this usually works even if scipy.stats fails

rho = np.corrcoef(
    np.argsort(np.argsort(S)),
    np.argsort(np.argsort(VS))
)[0,1]

print(f"Spearman ρ = {rho:.4f}")

plt.figure(figsize=(5,5))
plt.scatter(np.argsort(np.argsort(S)),
            np.argsort(np.argsort(VS)),
            alpha=0.3)
plt.xlabel("Sensitivity rank")
plt.ylabel("Vulnerability rank")
plt.title("Rank correlation (Spearman)")
plt.tight_layout()
plt.savefig("corr.png")
plt.show()

