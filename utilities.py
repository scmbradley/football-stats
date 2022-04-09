"""Utilities for doing generating new columns for football predictions"""


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
