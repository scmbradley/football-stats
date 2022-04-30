"""Generate and score predictions from football data."""

from pathlib import Path
import pandas as pd
import utilities


clean_in = Path("england_clean.csv")
with open(clean_in) as d:
    _df = pd.read_csv(d, keep_default_na=False)


print("Reminder: for log score, bigger is better, 0 is best.")
print("For Brier score, smaller is better, 0 is best.")
print("\n")

# Remove the early games of the season since their history-based predictions are
# all over the place.
# Not strictly necessary, but it also makes log scores behave better...

MAX_HISTORY = 6

home_len = _df["home_history"].str.len()
away_len = _df["away_history"].str.len()

df = _df[(home_len >= MAX_HISTORY) & (away_len >= MAX_HISTORY)].copy()


# First prediction method:
# Predict based on the proportion of home teams with this 5-game history
# who go on to win the game.

# Do this for home team and away team separately.


# Create dummy variables for the categorical H/D/A result

results_bools = (
    df["result"]
    .str.get_dummies()
    .rename(columns={"A": "home_loss", "D": "draw", "H": "home_win"})
)

df = pd.concat([df, results_bools], axis=1)
df_train = df[df["Season"] < 2018]
df_test = df[df["Season"] >= 2018]
results_train = df_train[["home_loss", "draw", "home_win"]]
results_test = df_test[["home_loss", "draw", "home_win"]]

# create a list of lists to use in graphing the data

score_list = []

for n in range(1, MAX_HISTORY + 1):
    score_list += utilities.create_form_scores(df_train, df_test, n)

# The above yields predictions based on form
# but which also "knows" about whether the team is home or away.

# To fix this, we need a weighted average of the two pivot tables,
# weighted by how often that form is encountered with the home/away team.

# Second prediction method:
# Predict based on average win rate for home team

home_team_average = results_train.mean()

# Acquire a frame of the right shape, and then set the columns to constants.
home_team_average_prediction = results_test.copy()


home_team_average_prediction[["home_loss", "draw", "home_win"]] = home_team_average[
    ["home_loss", "draw", "home_win"]
]

score_list.append(
    utilities.gen_score_list(
        home_team_average_prediction, results_test, "Home advantage"
    )
)

# Third prediction method
# What if you don't know which team is the home team?
# Average home win/home loss columns

home_away_average_prob = home_team_average[["home_loss", "home_win"]].mean()
draw_prob = home_team_average["draw"]

home_away_average_prediction = results_test.copy()
home_away_average_prediction["draw"] = draw_prob
home_away_average_prediction[["home_loss", "home_win"]] = home_away_average_prob


score_list.append(
    utilities.gen_score_list(home_away_average_prediction, results_test, "Win/draw")
)


# Fourth prediction method:
# literally just always guess one third.
thirds_prediction = results_test.copy()
thirds_prediction[["home_loss", "draw", "home_win"]] = 1 / 3

score_list.append(utilities.gen_score_list(thirds_prediction, results_test, "Thirds"))

# Predict using pivot table as before, but with a column of
# 3 games from home and 3 games from away team.

for n in [1, 2, 3]:
    score_list.append(utilities.create_both_form_scores(df_train, df_test, n))


# Final method: look to the odds.

odds_data = Path("odds_clean.csv")
with open(odds_data) as d:
    odds_df = pd.read_csv(d, keep_default_na=False)


results = (
    odds_df["FTR"]
    .str.get_dummies()
    .rename(columns={"A": "home_loss", "D": "draw", "H": "home_win"})
)

predictions = odds_df[["ProbH", "ProbA", "ProbD"]].rename(
    columns={"ProbH": "home_win", "ProbA": "home_loss", "ProbD": "draw"}
)

score_list.append(utilities.gen_score_list(predictions, results, "Odds"))


sl = pd.DataFrame(score_list, columns=["type", "log_score", "brier_score"])
sl.set_index("type", inplace=True)

# Normalise log and brier scores so that 1 is best and 0 is worst.

sl["log_norm"] = utilities.normalise_column(sl["log_score"])
sl["brier_norm"] = 1 - utilities.normalise_column(sl["brier_score"])

score_frame_out = Path("score_frame.csv")
sl.to_csv(score_frame_out)


# sl.sort_values("log_norm")[["log_norm", "brier_norm"]].plot.barh()
# plt.show()
