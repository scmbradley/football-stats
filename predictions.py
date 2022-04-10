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

_df["len_hist"] = _df.apply(lambda x: len(x["home_history"]), axis=1)

df = _df[_df["len_hist"] >= 5].copy()


# First prediction method:
# Predict based on the proportion of home teams with this 5-game history
# who go on to win the game.


df["past_five_home"] = df["home_history"].str[-5:]

# Create dummy variables for the categorical H/D/A result

results_bools = (
    df["result"]
    .str.get_dummies()
    .rename(columns={"A": "home_loss", "D": "draw", "H": "home_win"})
)

df = pd.concat([df, results_bools], axis=1)

# Use the fact that pivot table defaults to mean, to extract win/lose/draw
# probabilities for each five game history.

probs_frame = df.pivot_table(
    index="past_five_home", values=["home_loss", "draw", "home_win"]
)

# Use the columns of the pivot table as dictionaries to map
# 5-game histories to win/loss/draw probabilities

hw = df["past_five_home"].map(probs_frame["home_win"])
draw = df["past_five_home"].map(probs_frame["draw"])
hl = df["past_five_home"].map(probs_frame["home_loss"])

# Concat the above series into a frame that embodies the prediction

past_five_prediction = pd.concat(
    [hw.rename("home_win"), draw.rename("draw"), hl.rename("home_loss")], axis=1
)

# To make things easier, call a frame with the above shape a prediction frame
# And call results_bools a results frame


utilities.print_scores(past_five_prediction, results_bools, "Five score history")

# Second prediction method:
# Predict based on average win rate for home team

home_team_average = results_bools.mean()

# Acquire a frame of the right shape, and then set the columns to constants.
home_team_average_prediction = results_bools.copy()


home_team_average_prediction[["home_loss", "draw", "home_win"]] = home_team_average[
    ["home_loss", "draw", "home_win"]
]

utilities.print_scores(home_team_average_prediction, results_bools, "Home team average")

# Third prediction method
# What if you don't know which team is the home team?
# Average home win/home loss columns

home_away_average_prob = home_team_average[["home_loss", "home_win"]].mean()
draw_prob = home_team_average["draw"]

home_away_average_prediction = results_bools.copy()
home_away_average_prediction["draw"] = draw_prob
home_away_average_prediction[["home_loss", "home_win"]] = home_away_average_prob


utilities.print_scores(home_away_average_prediction, results_bools, "Home/Away average")

thirds_prediction = results_bools.copy()
thirds_prediction[["home_loss", "draw", "home_win"]] = 1 / 3

utilities.print_scores(thirds_prediction, results_bools, "Thirds")
