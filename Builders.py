import pandas as pd
import numpy as np
import math
from xgboost import XGBRegressor
from sklearn.decomposition import PCA


def build_difficulties_table(team_fdr_df, team_fixt_df, teams_names_dict, scores_df, history_df):
    # A dict that maps winning prob to game's difficulty
    prob_diff_dict = {(0, 0.2): 5.0, (0.2, 0.4): 4.0, (0.4, 0.6): 3.0, (0.6, 0.8): 2.0, (0.8, 1): 1.0}
    difficulties_df = team_fdr_df.copy()

    # Difficulty prediction for each team and game based on heuristic
    for i, gw in enumerate(team_fixt_df.columns):
        calculated_teams = []
        for team, nxt_games in team_fixt_df.iterrows():
            if team in calculated_teams:
                continue
            if nxt_games.iloc[i] == "BLANK":
                continue
            
            # Fetch team's data
            full_team_name = teams_names_dict[team]
            team_players_score = scores_df[scores_df["team"] == full_team_name]
            team_players_avg_score = team_players_score[f"GW {gw}"].mean()
            
            # Fetch opponent team's data
            opp_name = nxt_games.iloc[i].split('(')[0][:-1]
            full_opp_name = teams_names_dict[opp_name]
            opp_team_players_score = scores_df[scores_df["team"] == full_opp_name]
            opp_team_players_avg_score = opp_team_players_score[f"GW {gw}"].mean()
            calculated_teams.append(opp_name) # To prevent redundant calculations in loop

            # Extract difficulties defined by FPL
            team_diff = team_fdr_df.loc[team, gw]
            opp_team_diff = team_fdr_df.loc[opp_name, gw]

            # Fetch previous encounters between two teams
            head2head_df = history_df[(history_df["team_x"] == full_team_name) & (history_df["opp_team_name"] == full_opp_name)]
            head2head_df = head2head_df[["season_x", "team_x", "opp_team_name", "team_h_score", "team_a_score", "was_home", "GW"]].drop_duplicates()
            head2head_df = head2head_df.dropna(subset=["team_h_score", "team_a_score"])
            team_wins = 0
            opp_team_wins = 0

            # Calculate wins and loses stats between two teams
            for j in range(len(head2head_df)):
                was_home = head2head_df.iloc[j, head2head_df.columns.get_loc("was_home")]
                home_score = head2head_df.iloc[j, head2head_df.columns.get_loc("team_h_score")]
                away_score = head2head_df.iloc[j, head2head_df.columns.get_loc("team_a_score")]
                
                if home_score == away_score:
                    continue
                
                if was_home:
                    if home_score > away_score:
                        team_wins += 1
                    else:
                        opp_team_wins += 1
                else:
                    if home_score > away_score:
                        opp_team_wins += 1
                    else:
                        team_wins += 1

            # Calculate winning prob of both teams using softmax and based on features collected
            team_exp = math.exp(team_players_avg_score + team_wins - team_diff)
            opp_exp = math.exp(opp_team_players_avg_score + opp_team_wins - opp_team_diff)
            team_win_prob = team_exp / (team_exp + opp_exp)
            opp_win_prob = 1 - team_win_prob
            
            # Map prob to updated difficulty and build df
            for interval in prob_diff_dict.keys():
                if interval[0] < team_win_prob <= interval[1]:
                    difficulties_df.loc[team, gw] = prob_diff_dict[interval]
                if interval[0] < opp_win_prob <= interval[1]:
                    difficulties_df.loc[opp_name, gw] = prob_diff_dict[interval]
            
            return difficulties_df


def build_history(curr_season_df, last_two_seasons):
    # Merge last_two_seasons and curr_season_df
    history_df = pd.concat([last_two_seasons, curr_season_df])

    # Replace "GKP" with "GK" in the "position" column
    history_df["position"] = history_df["position"].apply(lambda x: "GK" if x == "GKP" else x)

    return history_df


def build_scores_table(curr_season_df, upcoming_fixt_df, name_team_dict, history_df, id_teams_dict, w2v_model):
    # Define columns for scores_df
    scores_df_cols = ["id", "name", "team"] + [f"GW {i}" for i in range(1, 39)]

    # Initialize scores_df with zeros
    scores_df = pd.DataFrame(0.0, columns=scores_df_cols, index=range(len(curr_season_df["element"].unique())))

    # Convert id, name, and team columns to appropriate data types
    scores_df[["id", "name", "team"]] = scores_df[["id", "name", "team"]].astype({"id": int, "name": str, "team": str})

    # Initialize an empty list to store future fixture rows
    future_fixt_rows = []

    # Iterate over unique player ids
    for i, player_id in enumerate(upcoming_fixt_df["p_id"].unique()):
        # Filter fixtures for the current player and sort by event
        p_fixt_df = upcoming_fixt_df[upcoming_fixt_df["p_id"] == player_id].sort_values("event", ascending=True)
        p_name = p_fixt_df["p_name"].iloc[0]  # Get player name
        p_team = name_team_dict[p_name]  # Get player's team name
        scores_df.loc[i, "name"] = p_name
        scores_df.loc[i, "id"] = player_id
        scores_df.loc[i, "team"] = p_team

        # Filter history data for the current player and sort by season and gameweek
        p_hist_df = history_df[history_df["element"] == player_id].sort_values(by=["season_x", "GW"], ascending=[False, False])
        
        # Filter current season data for the current player and sort by season and gameweek
        p_curr_df = curr_season_df[curr_season_df["element"] == player_id].sort_values(by=["season_x", "GW"], ascending=[False, False])
        
        # Get score of each gameweek in the current season
        game_weeks = p_curr_df["GW"].max()
        for j in range(1, game_weeks + 1):
            mask = p_curr_df["GW"] == j
            if mask.any():
                scores_df.loc[i, f"GW {j}"] = p_curr_df.loc[mask, "value"].mean() / 10
        
        # Iterate over future fixtures for the current player
        for _, fixture_row in p_fixt_df.iterrows():
            if fixture_row["provisional_start_time"]:
                continue
            
            # Extract relevant data for future fixture row
            row = p_hist_df.head(1).drop(columns=["value"]).copy()
            row["GW"] = int(fixture_row["event"])
            row["kickoff_time"] = fixture_row["kickoff_time"]
            row["was_home"] = bool(fixture_row["is_home"])
            
            if fixture_row["is_home"]:
                row["opp_team_name"] = id_teams_dict[fixture_row["team_a"]]
            else:
                row["opp_team_name"] = id_teams_dict[fixture_row["team_h"]]
            
            # Call build_for_pred function to create row for prediction
            row_to_pred = build_for_pred(row, p_hist_df)
            p_hist_df = pd.concat([p_hist_df, row])
            future_fixt_rows.append(row_to_pred)

    # Concatenate all future fixture rows into a DataFrame
    players_future_fixt_df = pd.concat(future_fixt_rows)

    # Build a train df from last two seasons and games that were played from current season
    final_fit_df = preprocess_data(history_df, w2v_model)
    y_fit = final_fit_df["value"]
    final_fit_df = final_fit_df.drop(columns=["value"])
    final_fit_df[final_fit_df.columns] = final_fit_df[final_fit_df.columns].astype(float)

    # Initialize model and train
    xgb_model = XGBRegressor()
    xgb_model.fit(final_fit_df, y_fit)
    players_future_fixt_df = players_future_fixt_df.dropna(subset=["team_h_score", "team_a_score"])

    # Build validation df and predict player's scores in future games
    final_validation_df = preprocess_data(players_future_fixt_df, w2v_model)
    final_validation_df[["GW", "was_home", "team_h_score", "team_a_score"]] = final_validation_df[["GW", "was_home", "team_h_score", "team_a_score"]]\
                                                                            .astype({"GW":int, "was_home":bool, "team_h_score":int, "team_a_score":int})
    val_preds = xgb_model.predict(final_validation_df)

    # Prepare scores to be filled in scores df
    pred_scores = np.round(val_preds / 10, decimals=1)
    pred_scores_df = pd.DataFrame(pred_scores, columns=["score"])
    players_future_fixt_df.reset_index(drop=True, inplace=True)
    pred_scores_df.reset_index(drop=True, inplace=True)
    players_with_score = pd.concat([players_future_fixt_df, pred_scores_df], axis=1)

    # Fill future games player scores in scores df
    for idx, id in enumerate(upcoming_fixt_df["p_id"].unique()):
        p_scores_df = players_with_score[players_with_score["element"] == id].sort_values("GW", ascending=True)
        for i in range(len(p_scores_df)):
            gw = p_scores_df.iloc[i, p_scores_df.columns.get_loc("GW")]
            scores_df.loc[idx, f"GW {gw}"] = p_scores_df.iloc[i, p_scores_df.columns.get_loc("score")]
    
    return scores_df


# Function that extracts data from a match
def extract_data(match_data, p_id, p_team, p_pos, p_name, season_x, id_teams_dict):
    return [
        season_x,
        p_name,
        p_pos,
        p_team,
        match_data.get("assists", None),
        match_data.get("bonus", None),
        match_data.get("bps", None),
        match_data.get("clean_sheets", None),
        match_data.get("creativity", None),
        p_id,
        match_data.get("fixture", None),
        match_data.get("goals_conceded", None),
        match_data.get("goals_scored", None),
        match_data.get("ict_index", None),
        match_data.get("influence", None),
        match_data.get("kickoff_time", None),
        match_data.get("minutes", None),
        match_data.get("opponent_team", None),
        id_teams_dict.get(match_data.get("opponent_team", None), None),
        match_data.get("own_goals", None),
        match_data.get("penalties_missed", None),
        match_data.get("penalties_saved", None),
        match_data.get("red_cards", None),
        match_data.get("round", None),
        match_data.get("saves", None),
        match_data.get("selected", None),
        match_data.get("team_a_score", None),
        match_data.get("team_h_score", None),
        match_data.get("threat", None),
        match_data.get("total_points", None),
        match_data.get("transfers_balance", None),
        match_data.get("transfers_in", None),
        match_data.get("transfers_out", None),
        match_data.get("value", None),
        match_data.get("was_home", None),
        match_data.get("yellow_cards", None),
        match_data.get("round", None)
    ]


def build_curr_season(curr_season_jasons, id_name_dict, name_team_dict, name_pos_dict, id_teams_dict):
    # Dict that maps position ID to position name
    num_to_string_pos = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    season_x = "2023-24"
    rows = []

    # Build rows that represent players' games
    for player_num in range(curr_season_jasons.shape[0]):
        p_id = curr_season_jasons.iloc[player_num, 0]["element"]
        p_name = id_name_dict[p_id]
        p_team = name_team_dict[p_name]
        p_pos = num_to_string_pos[name_pos_dict[p_name]]
        
        for gameweek_num in range(curr_season_jasons.shape[1]):
            match_data = curr_season_jasons.iloc[player_num, gameweek_num]
            if match_data is None:
                continue
            rows.append(extract_data(match_data, p_id, p_team, p_pos, p_name, season_x, id_teams_dict))

    # Current season df columns
    curr_season_df_cols = ['season_x', 'name', 'position', 'team_x', 'assists', 'bonus', 'bps',
        'clean_sheets', 'creativity', 'element', 'fixture', 'goals_conceded',
        'goals_scored', 'ict_index', 'influence', 'kickoff_time', 'minutes',
        'opponent_team', 'opp_team_name', 'own_goals', 'penalties_missed',
        'penalties_saved', 'red_cards', 'round', 'saves', 'selected',
        'team_a_score', 'team_h_score', 'threat', 'total_points',
        'transfers_balance', 'transfers_in', 'transfers_out', 'value',
        'was_home', 'yellow_cards', 'GW']

    # Build a df from rows
    curr_season_df = pd.DataFrame(rows, columns=curr_season_df_cols)

    # Drop rows with missing team scores
    curr_season_df.dropna(subset=["team_a_score", "team_h_score"], inplace=True)

    # Sort DataFrame
    curr_season_df.sort_values('GW', inplace=True)

    return curr_season_df


# A function that converts players names to vector representations
def name_to_vector(name, embedding_model):

    # Split the name into individual words
    words = name.split()

    # Initialize an empty vector
    vector = np.zeros(embedding_model.vector_size)

    # Number of words found in model count - for vector normalization
    normalizer = 0

    # Iterate through each word in the name and add its vector representation
    for word in words:
        if word in embedding_model:
            normalizer += 1
            vector += embedding_model[word]
            
    # Normalize the vector and reshape
    if normalizer > 0:
        vector /= normalizer
    vector = vector.reshape(1, -1)

    return vector


# A function that embeds textual features, reduce their dimension with PCA, and make each vector component a column in the dataframe
def text_to_vec(text_features, df, embedding_model):
    first_iter = True
    pca = PCA(n_components=25)
    for feature in text_features:
        df[feature] = df[feature].apply(lambda x: name_to_vector(x, embedding_model))
        vectors = np.stack(df[feature].to_numpy())
        vectors = np.reshape(vectors, (vectors.shape[0], vectors.shape[2])) # From 3 dimensions (x, 1, y) to 2 (x, y)
        if first_iter: # Fit once, only transform afterwards
            reduced_vectors = pca.fit_transform(vectors)
            first_iter = False
        reduced_vectors = pca.transform(vectors)
        df[feature] = list(reduced_vectors) # Each cell in column contains the whole vector as list
        column_name_mapping = dict(zip([i for i in range(25)], [f"{feature}_comp_{i}" for i in range(25)]))
        exploded_df = df[feature].apply(pd.Series) # Explode lists and make a column from each component
        exploded_df = exploded_df.rename(columns=column_name_mapping)
        df = pd.concat([df, exploded_df], axis=1) # Add new columns to original df

    return df   


def preprocess_data(df, embedding_model):
    # One-hot encoding position
    onehot_df = pd.get_dummies(df, columns=["position"])

    # cols to drop: transfers in & out (because balance = in - out), fixture, element, round (same as GW), name, season
    onehot_df = onehot_df.drop(columns=["season_x", "name", "element", "transfers_in", "transfers_out", "fixture", "round", "opponent_team"])

    # Converting string date to date time object and then to an integer time stamp divided by the number of nano-seconds in a second
    onehot_df["kickoff_time"] = pd.to_datetime(onehot_df["kickoff_time"]).astype('int64').div(10**9)

    # Convert textual features to embedded vectors and create finalize train dataset preparations
    textual_features = ["team_x", "opp_team_name"]
    onehot_df = text_to_vec(textual_features, onehot_df, embedding_model)
    preprocessed_df = onehot_df.drop(columns=textual_features) # Drop original columns after embedding
    
    return preprocessed_df


# Prepare future games data for prediction
def build_for_pred(validation_df, history_df):
    history_df = history_df.sort_values(by=["season_x", "GW"], ascending=[False, False])

    # Features to be filled by average of last 3 games
    future_features = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded', 'goals_scored', 'ict_index', 'influence', 
                       'minutes', 'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards', 'saves', 'selected', 'threat', 'total_points', 
                       'transfers_balance', 'yellow_cards']
    
    validation_df[future_features] = validation_df[future_features].astype(float)
    history_df[future_features] = history_df[future_features].astype(float)
    
    for i in range(len(validation_df)):
        # Fetch player's data
        p_name = validation_df.iloc[i, validation_df.columns.get_loc("name")]
        p_team = validation_df.iloc[i, validation_df.columns.get_loc("team_x")]
        opp_team = validation_df.iloc[i, validation_df.columns.get_loc("opp_team_name")]

        # Fetch dfs of player's previous games and stats about previous matches between his team and other teams from last 3 seasons
        prev_games = history_df[history_df["name"] == p_name]
        head2head_df = history_df[(history_df["team_x"] == p_team) & (history_df["opp_team_name"] == opp_team)]
        head2head_df = head2head_df[["season_x", "team_x", "opp_team_name", "team_h_score", "team_a_score", "was_home", "GW"]].drop_duplicates()
        p_team_score = []
        opp_team_score = []
        home_game = validation_df.iloc[i, validation_df.columns.get_loc("was_home")]

        if len(head2head_df) > 0:
            if len(head2head_df) > 3:
                head2head_df = head2head_df.head(3) # Look at last 3 encounters between teams
            
            # Gather amount of goals each team scored in all 3 games
            for j in range(len(head2head_df)):
                if head2head_df.iloc[j, head2head_df.columns.get_loc("was_home")]:
                    p_team_score.append(head2head_df.iloc[j, head2head_df.columns.get_loc("team_h_score")])
                    opp_team_score.append(head2head_df.iloc[j, head2head_df.columns.get_loc("team_a_score")])
                else:
                    p_team_score.append(head2head_df.iloc[j, head2head_df.columns.get_loc("team_a_score")])
                    opp_team_score.append(head2head_df.iloc[j, head2head_df.columns.get_loc("team_h_score")])
            
            # Predict the upcoming game's finak score to be the average of the goals each team scored in last 3 encounters
            if home_game:
                validation_df.iloc[i, validation_df.columns.get_loc("team_h_score")] = sum(p_team_score) / len(p_team_score)
                validation_df.iloc[i, validation_df.columns.get_loc("team_a_score")] = sum(opp_team_score) / len(opp_team_score)
            else:
                validation_df.iloc[i, validation_df.columns.get_loc("team_h_score")] = sum(opp_team_score) / len(opp_team_score)
                validation_df.iloc[i, validation_df.columns.get_loc("team_a_score")] = sum(p_team_score) / len(p_team_score)
        else: # Teams never played againt each other, so predcit a zero tie
            validation_df.iloc[i, validation_df.columns.get_loc("team_h_score")] = 0.0
            validation_df.iloc[i, validation_df.columns.get_loc("team_a_score")] = 0.0

        if len(prev_games) > 3:
            prev_games = prev_games.head(3) # Look at past 3 games the player played

        # Predict features' values by averaging on last 3 games
        for future_feature in future_features:
            validation_df.iloc[i, validation_df.columns.get_loc(future_feature)] = prev_games[future_feature].mean() if len(prev_games) > 0 else 0
    
    return validation_df