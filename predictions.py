from pathlib import Path
import pandas
from numpy import log2
from utilities import prediction_from_history

clean_in = Path("england_clean.csv")
with open(clean_in) as d:
    _df = pandas.read_csv(d, keep_default_na=False)


def predictor_tuple(df_counter):
    try:
        h = df_counter["H"]
    except KeyError:
        h = 0
    try:
        a = df_counter["A"]
    except KeyError:
        a = 0
    try:
        d = df_counter["D"]
    except KeyError:
        d = 0

    return h, a, d


_df["len_hist"] = _df.apply(lambda x: len(x["home_history"]), axis=1)

df = _df[_df["len_hist"] >= 5]

p_tup_dict = {"H": 0, "A": 1, "D": 2}


df["past_five_home"] = df.apply(lambda x: x["home_history"][-5:], axis=1)

crude_predictor = {}
for history in df["past_five_home"].unique():
    vc = df[df["past_five_home"] == history]["result"].value_counts()
    crude_predictor[history] = predictor_tuple(vc)

home_predictions_five = df.apply(lambda x: crude_predictor[x["past_five_home"]], axis=1)
df["home_predictions_five"] = home_predictions_five

home_predictions_full = df.apply(
    lambda x: prediction_from_history(x["home_history"]), axis=1
)
df["home_predictions_full"] = home_predictions_full


def probability_for_outcome(prediction, outcome):
    return prediction[p_tup_dict[outcome]] / sum(prediction)


# Using the log score because it's easy to implement
# Not really, I just want to annoy Richard Pettigrew
def score_prediction(prediction, result):
    """
    Scores a prediction based on a result.

    prediction: a tuple of (H,A,D) where each of H,A,D is an int
    result: one of H,A,D.
    """
    return log2(probability_for_outcome(prediction, result))


def score_prediction_brier(prediction, result):
    h_prob = probability_for_outcome(prediction, "H")
    a_prob = probability_for_outcome(prediction, "A")
    d_prob = probability_for_outcome(prediction, "D")
    return (
        ((result == "H") - h_prob) ** 2
        + ((result == "A") - a_prob) ** 2
        + ((result == "D") - d_prob) ** 2
    )


# Predict based just on number of home wins/ home draws/ home losses
climatology = predictor_tuple(df["result"].value_counts())


five_game_prediction_scores = df.apply(
    lambda x: score_prediction(x["home_predictions_five"], x["result"]),
    axis=1,
)

full_prediction_scores = df.apply(
    lambda x: score_prediction(x["home_predictions_full"], x["result"]),
    axis=1,
)

climatology_prediction_scores = df.apply(
    lambda x: score_prediction(climatology, x["result"]),
    axis=1,
)

third_scores = df.apply(
    lambda x: score_prediction((1, 1, 1), x["result"]),
    axis=1,
)

climatology_prediction_scores_brier = df.apply(
    lambda x: score_prediction_brier(climatology, x["result"]),
    axis=1,
)

five_game_prediction_scores_brier = df.apply(
    lambda x: score_prediction_brier(x["home_predictions_five"], x["result"]),
    axis=1,
)

full_prediction_scores_brier = df.apply(
    lambda x: score_prediction_brier(x["home_predictions_full"], x["result"]),
    axis=1,
)
