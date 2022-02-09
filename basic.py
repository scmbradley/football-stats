from pathlib import Path
import requests
import pandas


# Download data if it doesn't exist, or if forced

data_url = "https://raw.githubusercontent.com/jalapic/engsoccerdata/master/data-raw/england.csv"

data_raw = Path("england.csv")

force = False

if force or not data_raw.is_file():
    download = requests.get(data_url)
    data_raw.write_text(download.content.decode("utf-8"))

with open(data_raw) as d:
    data = pandas.read_csv(d)

# Focus on 20 team premier league

data_f = data[(data["Season"] >= 1995) & (data["tier"] == 1)]

df = data_f[["Date", "Season", "home", "visitor", "result"]].copy()


# Helper functions for extracting a team's games for a season


def games_with_team(name, frame=df):
    return frame[(frame["home"] == name) | (frame["visitor"] == name)]


def games_in_season(season, frame=df, before=None):
    season = frame[frame["Season"] == season]
    if before is not None:
        season = season[season["Date"] < before]
    return season


def get_history_frame(team, season, before=None, frame=df):
    with_team = games_with_team(team, frame=frame)
    in_season = games_in_season(season, frame=with_team, before=before)
    return in_season


# Add columns for a teams results so far

home_result = {"H": "W", "D": "D", "A": "L"}
away_result = {"H": "L", "D": "D", "A": "W"}


def game_to_result(game, team):
    if game["home"] == team:
        return home_result[game["result"]]
    else:
        return away_result[game["result"]]


def get_history_string(team, season, before=None, frame=df):
    hist_frame = get_history_frame(team, season, before, frame)
    result_series = hist_frame.apply(lambda x: game_to_result(x, team), axis=1)
    return "".join(result_series.values.tolist())


print("home history")
df["home_history"] = df.apply(
    lambda x: get_history_string(x["home"], x["Season"], before=x["Date"]), axis=1
)


print("away history")
df["away_history"] = df.apply(
    lambda x: get_history_string(x["visitor"], x["Season"], before=x["Date"]), axis=1
)


points_dict = {"W": 3, "D": 1, "L": 0}


def points_from_history(s):
    return sum([points_dict[x] for x in s])


print("home_points")
df["home_points"] = df.apply(lambda x: points_from_history(x["home_history"]), axis=1)

print("away points")
df["away_points"] = df.apply(lambda x: points_from_history(x["away_history"]), axis=1)

# It will be useful later to extract the last five games, for example, from the full series
def last_n_games(frame, n, away=False):
    if away:
        column = "away_history"
    else:
        column = "home_history"
    return frame[column].values.tolist()[-n]


# Also useful, as a comparison, the final record of results and points for a season


def get_full_season(team, season):
    hf = get_history_frame(team, season)
    return hf.values.tolist()[-1]["home_history"]


def get_final_points(team, season):
    return points_from_history(get_full_season(team, season))


clean_out = Path("england_clean.csv")
df.to_csv(clean_out)

print("Done")
