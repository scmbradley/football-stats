"""Utilities for doing generating new columns for football predictions"""


def get_teams_in_frame(frame):
    return frame["home"].unique()


# Helper functions for extracting a team's games for a season


def games_with_team(name, frame):
    return frame[(frame["home"] == name) | (frame["visitor"] == name)]


def games_in_season(season, frame, before=None, tier=None):
    season = frame[frame["Season"] == season]
    if before is not None:
        season = season[season["Date"] < before]
    if tier is not None:
        season = season[season["tier"] == tier]
    return season


def get_history_frame(team, season, frame, before=None):
    with_team = games_with_team(team, frame)
    in_season = games_in_season(season, with_team, before=before)
    return in_season


# Add columns for a teams results so far

home_result = {"H": "W", "D": "D", "A": "L"}
away_result = {"H": "L", "D": "D", "A": "W"}


def game_to_result(game, team):
    if game["home"] == team:
        return home_result[game["result"]]
    else:
        return away_result[game["result"]]


def get_history_string(team, season, frame, before=None):
    hist_frame = get_history_frame(team, season, frame, before)
    result_series = hist_frame.apply(lambda x: game_to_result(x, team), axis=1)
    return "".join(result_series.values.tolist())


points_dict = {"W": 3, "D": 1, "L": 0}


def points_from_history(s):
    return sum([points_dict[x] for x in s])


# It will be useful later to extract the last five games, for example, from the full series
def last_n_games(frame, n, away=False):
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


def get_final_points(team, season, frame):
    return points_from_history(get_full_season(team, season, frame))
