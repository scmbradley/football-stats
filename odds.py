import utilities
from pathlib import Path
import requests

data_url = "https://www.football-data.co.uk/mmz4281/2021/E0.csv"

data_raw = Path("odds.csv")

force = False

if force or not data_raw.is_file():
    download = requests.get(data_url)
    data_raw.write_text(download.content.decode("utf-8"))

with open(data_raw) as d:
    data = pandas.read_csv(d)

# Odds are represented here as, essentially, a multiplier on your stake.
# Described in football-data docs as "1X2" betting.
# So odds of 2 is an implied probability of 1/2.
df = data[["Date", "HomeTeam", "AwayTeam", "FTR", "AvgH", "AvgA", "AvgD"]].copy()

df["ProbH"] = 1 / df["AvgH"]
df["ProbA"] = 1 / df["AvgA"]
df["ProbD"] = 1 / df["AvgD"]