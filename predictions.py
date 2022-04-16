from pathlib import Path
import pandas as pd
import utilities
import matplotlib.pyplot as plt

clean_in = Path("england_clean.csv")
with open(clean_in) as d:
    _df = pd.read_csv(d, keep_default_na=False)


print("Reminder: for log score, bigger is better, 0 is best.")
print("For Brier score, smaller is better, 0 is best.")
print("\n")

# Remove the early games of the season since their history-based predictions are
# all over the place.
# Not strictly necessary, but it also makes log scores behave better...

home_len = _df["home_history"].str.len()
away_len = _df["away_history"].str.len()

df = _df[(home_len >= 5) & (away_len >= 5)].copy()


# First prediction method:
# Predict based on the proportion of home teams with this 5-game history
# who go on to win the game.

# Do this for home team and away team separately.


df["past_five_home"] = df["home_history"].str[-5:]
df["past_five_away"] = df["away_history"].str[-5:]

# Create dummy variables for the categorical H/D/A result

results_bools_home = (
    df["result"]
    .str.get_dummies()
    .rename(columns={"A": "home_loss", "D": "draw", "H": "home_win"})
)

df = pd.concat([df, results_bools_home], axis=1)

# note that away_win == home_loss, and away_loss == home_win.

# Use the fact that pivot table defaults to mean, to extract win/lose/draw
# probabilities for each five game history.

probs_frame_home = df.pivot_table(
    index="past_five_home", values=["home_loss", "draw", "home_win"]
)

probs_frame_away = df.pivot_table(
    index="past_five_away", values=["home_win", "draw", "home_loss"]
)

past_five_home_prediction = utilities.prob_frame_to_prediction(
    df, "past_five_home", probs_frame_home
)

past_five_away_prediction = utilities.prob_frame_to_prediction(
    df, "past_five_away", probs_frame_away
)

# create a list of lists to use in graphing the data

score_list = []

score_list.append(
    utilities.gen_score_list(
        past_five_home_prediction, results_bools_home, "Form (5, home)"
    )
)

score_list.append(
    utilities.gen_score_list(
        past_five_away_prediction, results_bools_home, "Form (5, away)"
    )
)


# The above yields predictions based on form
# but which also "knows" about whether the team is home or away.

# To fix this, we need a weighted average of the two pivot tables,
# weighted by how often that form is encountered with the home/away team.

home_counts = df["past_five_home"].value_counts()
away_counts = df["past_five_away"].value_counts()

weights = home_counts / (home_counts + away_counts)

probs_frame_unknown = probs_frame_home.mul(weights, axis=0) + probs_frame_away.mul(
    1 - weights, axis=0
)

# Concat the above series into a frame that embodies the prediction

past_five_unknown_prediction = utilities.prob_frame_to_prediction(
    df, "past_five_home", probs_frame_unknown
)


score_list.append(
    utilities.gen_score_list(
        past_five_unknown_prediction, results_bools_home, "Form (5, Unknown)"
    )
)

score_list += utilities.create_form_scores(df, 3)

# Second prediction method:
# Predict based on average win rate for home team

home_team_average = results_bools_home.mean()

# Acquire a frame of the right shape, and then set the columns to constants.
home_team_average_prediction = results_bools_home.copy()


home_team_average_prediction[["home_loss", "draw", "home_win"]] = home_team_average[
    ["home_loss", "draw", "home_win"]
]

score_list.append(
    utilities.gen_score_list(
        home_team_average_prediction, results_bools_home, "Home advantage"
    )
)

# Third prediction method
# What if you don't know which team is the home team?
# Average home win/home loss columns

home_away_average_prob = home_team_average[["home_loss", "home_win"]].mean()
draw_prob = home_team_average["draw"]

home_away_average_prediction = results_bools_home.copy()
home_away_average_prediction["draw"] = draw_prob
home_away_average_prediction[["home_loss", "home_win"]] = home_away_average_prob


score_list.append(
    utilities.gen_score_list(
        home_away_average_prediction, results_bools_home, "Win/draw"
    )
)

thirds_prediction = results_bools_home.copy()
thirds_prediction[["home_loss", "draw", "home_win"]] = 1 / 3

score_list.append(
    utilities.gen_score_list(thirds_prediction, results_bools_home, "Thirds")
)

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
