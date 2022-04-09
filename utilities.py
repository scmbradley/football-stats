"""Utilities for doing generating new columns for football predictions"""

from numpy import log2


def get_teams_in_frame(frame):
    """Returns a Series of names of teams in the frame."""
    return frame["home"].unique()


# Helper functions for extracting a team's games for a season


def games_with_team(name, frame):
    """Returns a Frame of games including named team."""
    return frame[(frame["home"] == name) | (frame["visitor"] == name)]


def games_in_season(season, frame, before=None, tier=None):
    """
    Returns a Frame including games from the specified season.

    Optional filtering by date (before) and tier (tier).
    """
    season = frame[frame["Season"] == season]
    if before is not None:
        season = season[season["Date"] < before]
    if tier is not None:
        season = season[season["tier"] == tier]
    return season


def get_history_frame(team, season, frame, before=None):
    """Return Frame containing specified team's games from season."""
    with_team = games_with_team(team, frame)
    in_season = games_in_season(season, with_team, before=before)
    return in_season


# Add columns for a teams results so far

home_result = {"H": "W", "D": "D", "A": "L"}
away_result = {"H": "L", "D": "D", "A": "W"}


def game_to_result(game, team):
    """
    Translate a game into a result for the specified team.

    Result given in the form of W/L/D.
    """
    if game["home"] == team:
        return home_result[game["result"]]
    else:
        return away_result[game["result"]]


def get_history_string(team, season, frame, before=None):
    """Get a string representing the historical performance of the team in the season to date."""
    hist_frame = get_history_frame(team, season, frame, before)
    result_series = hist_frame.apply(lambda x: game_to_result(x, team), axis=1)
    return "".join(result_series.values.tolist())


points_dict = {"W": 3, "D": 1, "L": 0}


# This is a pretty weird way to do things.
# I think I should revisit this whole approach.


def points_from_history(s):
    """Translate a history string into a number of points."""
    return sum([points_dict[x] for x in s])


# It will be useful later to extract the last five games, for example, from the full series
def last_n_games(frame, n, away=False):
    """Return the last n games of a history string."""
    if away:
        column = "away_history"
    else:
        column = "home_history"
    return frame[column].values.tolist()[-n]


# Also useful, as a comparison, the final record of results and points for a season


def get_full_season(team, season, frame):
    """Get full season as a string."""
    hf = get_history_frame(team, season, frame)
    return hf.values.tolist()[-1]["home_history"]


def prediction_from_history(history_string, home=True):
    """Return a prediction tuple based purely on the history."""
    w, l, d = 0, 0, 0
    for game in history_string:
        if game == "W":
            w += 1
        elif game == "L":
            l += 1
        elif game == "D":
            d += 1
    if home:
        return w, l, d
    else:
        return l, w, d


def get_final_points(team, season, frame):
    """Return team's final points tally from season."""
    return points_from_history(get_full_season(team, season, frame))


# functions for scoring predictions


def predictor_tuple(df_counter):
    """
    Return a predictor tuple based on a counter.

    A predictor tuple is just a tuple of numbers representing the relative probabilities of
    home win, away win and draw.
    """
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


# Where in the predictor tuple can I find `key` number?
p_tup_dict = {"H": 0, "A": 1, "D": 2}


def probability_for_outcome(prediction, outcome):
    """Translate a predictor tuple into a probability."""
    return prediction[p_tup_dict[outcome]] / sum(prediction)


# Using the log score because it's easy to implement
# Not really, I just want to annoy Richard Pettigrew
def score_prediction_log(prediction, result):
    """
    Scores a prediction based on a result.

    prediction: a tuple of (H,A,D) where each of H,A,D is an int
    result: one of H,A,D.
    """
    return log2(probability_for_outcome(prediction, result))


# OK fine I'll implement the Brier score too.
# Happy now, Richard?
def score_prediction_brier(prediction, result):
    h_prob = probability_for_outcome(prediction, "H")
    a_prob = probability_for_outcome(prediction, "A")
    d_prob = probability_for_outcome(prediction, "D")
    return (
        ((result == "H") - h_prob) ** 2
        + ((result == "A") - a_prob) ** 2
        + ((result == "D") - d_prob) ** 2
    )


def print_prediction(frame, prediction, title):
    scores = frame.apply(
        lambda x: score_prediction_log(prediction, x["result"]), axis=1
    )
