"""Generate graphs relating to the form predictions."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

score_frame_path = Path("score_frame.csv")
with open(score_frame_path) as d:
    sl = pd.read_csv(d)
sl.set_index("type", inplace=True)
sl.sort_values("log_norm")[["log_norm", "brier_norm"]].plot.barh()

plt.show()
