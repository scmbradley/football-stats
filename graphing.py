"""Generate graphs relating to the form predictions."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

score_frame_path = Path("score_frame.csv")
with open(score_frame_path) as d:
    sl = pd.read_csv(d)
sl.set_index("type", inplace=True)

sns.set_theme()
sns.set_style("white")

# Summary of all results.

sl.sort_values("log_norm")[["log_norm", "brier_norm"]].plot.barh()
# plt.xticks(rotation=45)
plt.legend(loc="lower right")
sns.despine(left=True)
plt.tight_layout()
plt.savefig("plots/summary.png")
plt.close("all")
# Line graph of score versus form length

home_values = ["Home advantage"] + [f"Form ({n}, home)" for n in range(1, 6)]
away_values = ["Home advantage"] + [f"Form ({n}, away)" for n in range(1, 6)]
unknown_values = ["Win/draw"] + [f"Form ({n}, unknown)" for n in range(1, 6)]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.8))


ax1.plot(sl["log_norm"][home_values].to_list(), label="Home")
ax1.plot(sl["log_norm"][away_values].to_list(), label="Away")
ax1.plot(sl["log_norm"][unknown_values].to_list(), label="Unknown")
ax2.plot(sl["brier_norm"][home_values].to_list())
ax2.plot(sl["brier_norm"][away_values].to_list())
ax2.plot(sl["brier_norm"][unknown_values].to_list())
ax1.set_title("Normalised log score")
ax2.set_title("Normalised Brier score")
plt.setp((ax1, ax2), xticks=np.arange(6), xticklabels=np.arange(6))
fig.suptitle("Scores for form predictions of length n")
fig.legend(loc=(0.08, 0.65))
fig.tight_layout()

plt.savefig("plots/score_form.png")
# plt.show()
plt.close("all")


# Line graph of score versus both form length

both_values = ["Home advantage"] + [f"Both form ({n})" for n in range(1, 4)]

fig, ax1 = plt.subplots(1, 1)


ax1.plot(sl["log_norm"][both_values].to_list(), label="Normalised log score")
ax1.plot(sl["brier_norm"][both_values].to_list(), label="Normalised Brier score")

plt.setp(ax1, xticks=np.arange(4), xticklabels=np.arange(4))
fig.suptitle("Scores for both form predictions of length n")
fig.tight_layout()
fig.legend(loc=(0.4, 0.15))
fig.tight_layout()
plt.savefig("plots/score_both_form.png")

# Line graph of score versus both form length

plt.close("all")
