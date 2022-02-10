from pathlib import Path
import pandas
from math import log2

clean_in = Path("england_clean.csv")
with open(clean_in) as d:
    df = pandas.read_csv(d, keep_default_na=False)


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


p_tup_dict = {"H": 0, "A": 1, "D": 2}

df["past_five_home"] = df.apply(lambda x: x["home_history"][-5:], axis=1)

crude_predictor = {}
for history in df["past_five_home"].unique():
    vc = df[df["past_five_home"] == history]["result"].value_counts()
    crude_predictor[history] = predictor_tuple(vc)

home_predictions = df.apply(lambda x: crude_predictor[x["past_five_home"]], axis=1)
df["home_predictions"] = home_predictions

# Could do this for away predictions, possibly better to make a predictor dict from away team histories
# away_predictions = df.apply(lambda x: crude_predictor[x["past_five_away"]], axis=1)
# df["away_predictions"] = away_predictions


# Using the log score because it's easy to implement
# Not really, I just want to annoy Richard Pettigrew
def score_prediction(prediction, result):
    """
    Scores a prediction based on a result.

    prediction: a tuple of (H,A,D) where each of H,A,D is an int
    result: one of H,A,D.
    """
    prob = prediction[p_tup_dict[result]] / sum(prediction)
    return log2(prob)


# Predict based just on number of home wins/ home draws/ home losses
climatology = predictor_tuple(df["result"].value_counts())


five_game_prediction_scores = df.apply(
    lambda x: score_prediction(x["home_predictions"], x["result"]),
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
