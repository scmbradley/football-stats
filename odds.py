import utilities
from pathlib import Path
import requests
import pandas as pd

data_url = "https://www.football-data.co.uk/mmz4281/2021/E0.csv"

odds_data_raw = Path("odds.csv")

force = False

if force or not odds_data_raw.is_file():
    download = requests.get(data_url)
    odds_data_raw.write_text(download.content.decode("utf-8"))

with open(odds_data_raw) as d:
    data = pd.read_csv(d)

# Odds are represented here as, essentially, a multiplier on your stake.
# Described in football-data docs as "1X2" betting.
# So odds of 2 is an implied probability of 1/2.
odds_df = data[["Date", "HomeTeam", "AwayTeam", "FTR", "AvgH", "AvgA", "AvgD"]].copy()

odds_df["ProbH"] = 1 / odds_df["AvgH"]
odds_df["ProbA"] = 1 / odds_df["AvgA"]
odds_df["ProbD"] = 1 / odds_df["AvgD"]

# Could also look at normalised probabilites, but I suspect the differences would be minimal.

# Write the useful information to a new file.
clean_out = Path("odds_clean.csv")
odds_df.to_csv(clean_out)


results = (
    odds_df["FTR"]
    .str.get_dummies()
    .rename(columns={"A": "home_loss", "D": "draw", "H": "home_win"})
)

predictions = odds_df[["ProbH", "ProbA", "ProbD"]].rename(
    columns={"ProbH": "home_win", "ProbA": "home_loss", "ProbD": "draw"}
)

utilities.print_scores(predictions, results, "Odds probabilities")
