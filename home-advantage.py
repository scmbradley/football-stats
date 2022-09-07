"""Graphing home advantage over time."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()
sns.set_style("white")


clean_in = Path("england_clean.csv")
with open(clean_in) as d:
    df = pd.read_csv(d, keep_default_na=False)

results_columns = ["home_loss", "draw", "home_win"]
results_bools = df[results_columns]

wins_per_season_avg = df.pivot_table(index="Season", values=results_columns).loc[
    :, results_columns
]
wins_per_season_tot = df.pivot_table(
    index="Season", values=results_columns, aggfunc=sum
)

ax = wins_per_season_avg.plot.area()
ax.legend(
    bbox_to_anchor=(0.2, -0.28, 0.6, 0.2), ncol=3, mode="expand", loc="upper left"
)
sns.despine(left=True, bottom=True)

# TODO: break down by tier
# TODO: line graph of home wins only
# TODO: rolling window version
plt.show()
