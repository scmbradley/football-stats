from pathlib import Path
import requests
import pandas

data_url = "https://raw.githubusercontent.com/jalapic/engsoccerdata/master/data-raw/england.csv"

data_raw = Path("england.csv")

if not data_raw.is_file():
    download = requests.get(data_url)
    data_raw.write_text(download.content.decode("utf-8"))

with open(data_raw) as d:
    data = pandas.read_csv(d)

# Focus on 20 team premier league

data_f = data[(data["Season"] >= 1995) & (data["tier"] == 1)]

df = data_f[["Date", "Season", "home", "visitor", "result"]].copy()


results_dict = {}

for team in df["home"].unique():
    results_dict[team] = ""


def last_five(result, previous):
    if len(previous) < 5:
        return previous + result
    else:
        return previous[-4:] + result


def make_history(team_name, result):
    previous_results = results_dict[team_name]
    updated_results = last_five(result, previous_results)
    results_dict[team_name] = updated_results
    return previous_results


home_result = {"H": "W", "D": "D", "A": "L"}
away_result = {"H": "L", "D": "D", "A": "W"}

pf_away = []
pf_home = []

# This is slow. Is there a better way?
for row in df.iterrows():
    r = row[1]
    home_name = r["home"]
    away_name = r["visitor"]
    h_result = home_result[r["result"]]
    a_result = away_result[r["result"]]
    pf_home.append(make_history(home_name, h_result))
    pf_away.append(make_history(away_name, a_result))

pf_home_3 = [x[-3:] for x in pf_home]
pf_away_3 = [x[-3:] for x in pf_away]

df["past_five_home"] = pf_home
df["past_five_away"] = pf_away
df["past_three_home"] = pf_home_3
df["past_three_away"] = pf_away_3

clean_out = Path("england_clean.csv")
df.to_csv(clean_out)
