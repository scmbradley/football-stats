from pathlib import Path
import pandas
import utilities

clean_in = Path("england_clean.csv")
with open(clean_in) as d:
    _df = pandas.read_csv(d, keep_default_na=False)


# Remove the early games of the season since their history-based predictions are
# all over the place.
# Not strictly necessary, but it also makes log scores behave better...

_df["len_hist"] = _df.apply(lambda x: len(x["home_history"]), axis=1)

df = _df[_df["len_hist"] >= 5]


# First prediction method:
# Predict based on the proportion of home teams with this 5-game history
# who go on to win the game.

df["past_five_home"] = df.apply(lambda x: x["home_history"][-5:], axis=1)

crude_predictor = {}
for history in df["past_five_home"].unique():
    vc = df[df["past_five_home"] == history]["result"].value_counts()
    crude_predictor[history] = utilities.predictor_tuple(vc)

home_predictions_five = df.apply(lambda x: crude_predictor[x["past_five_home"]], axis=1)
df["home_predictions_five"] = home_predictions_five

# Compare with predictions based on the home team's history over the whole season.

home_predictions_full = df.apply(
    lambda x: utilities.prediction_from_history(x["home_history"]), axis=1
)
df["home_predictions_full"] = home_predictions_full


# Predict based just on number of home wins/ home draws/ home losses
# That is, overall in the whole data frame.
climatology = utilities.predictor_tuple(df["result"].value_counts())

five_game_prediction_scores = df.apply(
    lambda x: utilities.score_prediction_log(x["home_predictions_five"], x["result"]),
    axis=1,
)

full_prediction_scores = df.apply(
    lambda x: utilities.score_prediction_log(x["home_predictions_full"], x["result"]),
    axis=1,
)

climatology_prediction_scores = df.apply(
    lambda x: utilities.score_prediction_log(climatology, x["result"]),
    axis=1,
)

third_scores = df.apply(
    lambda x: utilities.score_prediction_log((1, 1, 1), x["result"]),
    axis=1,
)

climatology_prediction_scores_brier = df.apply(
    lambda x: utilities.score_prediction_brier(climatology, x["result"]),
    axis=1,
)

five_game_prediction_scores_brier = df.apply(
    lambda x: utilities.score_prediction_brier(x["home_predictions_five"], x["result"]),
    axis=1,
)

full_prediction_scores_brier = df.apply(
    lambda x: utilities.score_prediction_brier(x["home_predictions_full"], x["result"]),
    axis=1,
)

third_scores_brier = df.apply(
    lambda x: utilities.score_prediction_brier((1, 1, 1), x["result"]),
    axis=1,
)


print("Reminder: for log score, bigger is better, 0 is best.")
print("For Brier score, smaller is better, 0 is best.")
print("\n")
print("Climatology prediction mean score.")
print(f"     Log: {climatology_prediction_scores.mean()}")
print(f"     Brier: {climatology_prediction_scores_brier.mean()}")

print("Five game prediction mean score.")
print(f"     Log: {five_game_prediction_scores.mean()}")
print(f"     Brier: {five_game_prediction_scores_brier.mean()}")

print("Thirds prediction mean score.")
print(f"     Log: {third_scores.mean()}")
print(f"     Brier: {third_scores_brier.mean()}")

print("Full season prediction mean score.")
print(f"     Log: {full_prediction_scores.mean()}")
print(f"     Brier: {full_prediction_scores_brier.mean()}")

# Puzzling that full prediction is *worse* then everything else.
# WHy!?
