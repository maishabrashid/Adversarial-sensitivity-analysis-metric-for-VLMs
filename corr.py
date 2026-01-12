import numpy as np
import matplotlib.pyplot as plt

vulnerability_scores = [20.94, 20.94, 20.93, 20.94, 20.94]
sensitivity_scores = [1.65, 1.65, 1.65, 1.65, 1.65]

#scores for PGD vs Vulnerabilty score
#vulnerability_scores = [14.47,11.25, 12.86,17.7]
#sensitivity_scores = [0.0241, 0.0242, 0.0220, 0.0341]

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

