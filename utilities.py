"""Utilities for generating new columns for football predictions."""

from numpy import log2
import pandas as pd


def get_teams_in_frame(frame):
    """Return a Series of names of teams in the frame."""
    return frame["home"].unique()


# Helper functions for extracting a team's games for a season


def games_with_team(name, frame):
    """Return a Frame of games including named team."""
    return frame[(frame["home"] == name) | (frame["visitor"] == name)]


def games_in_season(season, frame, before=None, tier=None):
    """
    Return a Frame including games from the specified season.

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
result_to_points_pre_1981 = {"W": 2, "D": 1, "L": 0}
result_to_points = {"W": 3, "D": 1, "L": 0}


def create_history_series(season):
    """Return home and away history series for given season."""
    home_histories = []
    away_histories = []
    for team in get_teams_in_frame(season):
        # Extract home and away matches for team
        home_results = season[season["home"] == team].copy()
        away_results = season[season["visitor"] == team].copy()

        # Translate A/D/H results into L/D/W for home and W/D/L for away
        home_results["team_result"] = home_results["result"].map(home_result)
        away_results["team_result"] = away_results["result"].map(away_result)

        # Put the season's matches back together
        team_results = pd.concat([home_results, away_results]).sort_index()

        # Get cumulative results so far (removing current match)
        team_results["form"] = team_results["team_result"].cumsum().str[:-1]
        team_results["points"] = (
            team_results["team_result"]
            .map(result_to_points)
            .shift(fill_value=0)
            .cumsum()
        )

        # Re-extract home and away series
        home_bool = team_results["home"] == team
        home_history = team_results[home_bool][["form", "points"]]
        away_history = team_results[~home_bool][["form", "points"]]

        # Add this team's form to the list of home/away forms
        home_histories.append(home_history)
        away_histories.append(away_history)

    # Turn the lists of forms into pandas series and then return the pair
    all_home = pd.concat(home_histories)
    all_away = pd.concat(away_histories)
    return all_home, all_away


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


# Also useful, as a comparison, the final record of results and points for a season


def get_full_season(team, season, frame):
    """Get full season as a string."""
    hf = get_history_frame(team, season, frame)
    return hf.values.tolist()[-1]["home_history"]


def score_log(prediction, result):
    """Return log score of predictions given results."""
    return log2((prediction * result).sum(axis=1))


def score_brier(prediction, result):
    """Return Brier score of prediction given results."""
    return ((result - prediction) ** 2).sum(axis=1)


def gen_score_list(prediction, result, title, printout=True):
    """Create score list for the score frame."""
    ret = [
        title,
        score_log(prediction, result).mean(),
        score_brier(prediction, result).mean(),
    ]
    if printout:
        print_score_list(ret)
    return ret


def print_score_list(in_list):
    """Print score list."""
    print(f"{in_list[0]} mean scores:")
    print(f"     Log score: {in_list[1]}")
    print(f"     Brier score: {in_list[2]}")


def print_scores(prediction, result, title):
    """Print scores of predictions given results."""
    print_score_list(gen_score_list(prediction, result, title))


def normalise_column(series):
    """Return min/max normalisation of a column."""
    return (series - series.min()) / (series.max() - series.min())


def prob_frame_to_prediction(game_frame, col_name, prob_frame):
    """
    Return a prediction frame based on the input game frame and prob frame.

    The col_name must be a data column whose entries match
    the keys to the prob_frame.
    """
    w = game_frame[col_name].map(prob_frame["home_win"])
    d = game_frame[col_name].map(prob_frame["draw"])
    l = game_frame[col_name].map(prob_frame["home_loss"])
    prediction = pd.concat(
        [w.rename("home_win"), d.rename("draw"), l.rename("home_loss")],
        axis=1,
    )
    return prediction


def create_form_scores(train_frame, test_frame, length):
    """
    Create lists of scores for home, away and unknown form of length length.

    Function does not check whether shorter game histories have been removed.
    """
    print("Generating form prediction scores for length " + str(length))
    # Copy the dataframe and generate form columns of the right length.

    values = ["home_loss", "draw", "home_win"]
    base_home_history = "home_history"
    base_away_history = "away_history"
    train = train_frame.copy()
    test = test_frame.copy()
    for frame in [test, train]:
        frame["form_home"] = frame[base_home_history].str[-length:]
        frame["form_away"] = frame[base_away_history].str[-length:]

    form_vc = train["form_home"].value_counts()
    print(
        f"Minimum value count {form_vc.min()}, length {len(form_vc)} out of {3**length}"
    )

    # Generate home and away form probabilities
    # Use the fact that pivot table defaults to mean, to extract win/lose/draw
    # probabilities for each game history.

    probs_frame_home = train.pivot_table(index="form_home", values=values)

    probs_frame_away = train.pivot_table(index="form_away", values=values)

    # Generate unknown form probabilities
    home_counts = train["form_home"].value_counts()
    away_counts = train["form_away"].value_counts()

    weights = home_counts / (home_counts + away_counts)

    probs_frame_unknown = probs_frame_home.mul(weights, axis=0) + probs_frame_away.mul(
        1 - weights, axis=0
    )

    # Generate predictions
    form_home_prediction = prob_frame_to_prediction(test, "form_home", probs_frame_home)

    form_away_prediction = prob_frame_to_prediction(test, "form_away", probs_frame_away)

    form_unknown_prediction = prob_frame_to_prediction(
        test, "form_home", probs_frame_unknown
    )

    # Extract results:
    results_bools = test[values]

    # Score predictions
    form_home_score = gen_score_list(
        form_home_prediction, results_bools, f"Form ({length}, home)", printout=False
    )

    form_away_score = gen_score_list(
        form_away_prediction, results_bools, f"Form ({length}, away)", printout=False
    )

    form_unknown_score = gen_score_list(
        form_unknown_prediction,
        results_bools,
        f"Form ({length}, unknown)",
        printout=False,
    )

    return [form_home_score, form_away_score, form_unknown_score]


def create_both_form_scores(train_frame, test_frame, length):
    """
    Create score_list using predictions from both teams.

    Even with the full dataframe, this will only really work up to
    length=3.
    """
    print("Generating form prediction scores for length " + str(length))
    # Copy the dataframe and generate form columns of the right length.
    values = ["home_loss", "draw", "home_win"]
    base_home_history = "home_history"
    base_away_history = "away_history"
    df_train = train_frame.copy()
    df_test = test_frame.copy()

    for frame in [df_train, df_test]:
        frame["both_form"] = (
            frame[base_home_history].str[-length:]
            + frame[base_away_history].str[-length:]
        )
    form_vc = df_train["both_form"].value_counts()
    print(
        f"Minimum value count {form_vc.min()}, length {len(form_vc)} out of {3**(length*2)}"
    )

    # Extract results:
    results_bools = df_test[values]

    # Generate probabilities
    both_form_probs = df_train.pivot_table(index="both_form", values=values)
    # Generate predictions
    both_form_prediction = prob_frame_to_prediction(
        df_test, "both_form", both_form_probs
    )

    return gen_score_list(
        both_form_prediction, results_bools, f"Both form ({length})", printout=False
    )
