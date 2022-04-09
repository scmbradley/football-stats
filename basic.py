from pathlib import Path
import requests
import pandas
import utilities
import time

# Download data if it doesn't exist, or if forced

data_url = "https://raw.githubusercontent.com/jalapic/engsoccerdata/master/data-raw/england.csv"

data_raw = Path("england.csv")

force = False

if force or not data_raw.is_file():
    download = requests.get(data_url)
    data_raw.write_text(download.content.decode("utf-8"))

with open(data_raw) as d:
    data = pandas.read_csv(d)

# To focus on 20 team premier league : season >= 1995, tier= 1
# uncomment the next line, and comment out the one after.

# data_f = data[(data["Season"] >= 1982) & (data["tier"] <= 2)]

data_f = data

# Remove a lot of extraneous columns.
df = data_f[["Date", "Season", "tier", "home", "visitor", "result"]].copy()


# We now add columns for the season-to-date win/loss history.
# This is quite slow, because it's using apply.
# We should figure out how to do the same with groupby and cumsum.
start = time.monotonic()
df_groups = df.groupby("Season")
print("Generating history")
history_groups = []
for year, season in df_groups:
    print(year)
    season["home_history"] = season.apply(
        lambda x: utilities.get_history_string(
            x["home"], x["Season"], season, before=x["Date"]
        ),
        axis=1,
    )
    season["away_history"] = season.apply(
        lambda x: utilities.get_history_string(
            x["visitor"], x["Season"], season, before=x["Date"]
        ),
        axis=1,
    )
    history_groups.append(season)

df["home_history"] = pandas.concat(history_groups)["home_history"]
df["away_history"] = pandas.concat(history_groups)["away_history"]


print("Generating points")
df["home_points"] = df.apply(
    lambda x: utilities.points_from_history(x["home_history"]), axis=1
)
df["away_points"] = df.apply(
    lambda x: utilities.points_from_history(x["away_history"]), axis=1
)

# Write the useful information to a new file.
clean_out = Path("england_clean.csv")
df.to_csv(clean_out)

print("Done")
print(f"Finished in {time.monotonic() - start}")
