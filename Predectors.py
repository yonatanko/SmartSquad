import pandas as pd
import gensim.downloader as api
from data_collection.fpl_api_collection import get_player_id_dict, get_name_and_pos_and_team_dict, get_bootstrap_data,\
                                               collate_player, get_fixture_dfs
from Builders import *


def get_scores_and_difficulties():
    # Dict that maps player's ID to player's name
    id_name_dict = get_player_id_dict()

    # Dicts that map player's name to player's position and team
    name_pos_dict, name_team_dict = get_name_and_pos_and_team_dict()

    # Get dict from FPL site that contains data about each team
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])

    # Dict that maps team's ID to team's name
    id_teams_dict = dict(zip(teams_df["id"], teams_df["name"]))

    # Dict that maps team's abbreviated name to team's full name
    teams_names_dict = dict(zip(teams_df["short_name"], teams_df["name"]))

    data_path = "Fantasy-Premier-Leaguue/data/cleaned_merged_seasons.csv"
    all_seasons_all_players_df = pd.read_csv(data_path)

    # Filter last two seasons from all previous seasons
    last_two_seasons = all_seasons_all_players_df[(all_seasons_all_players_df["season_x"] == "2022-23") | 
                                                (all_seasons_all_players_df["season_x"] == "2021-22")]
    # FPL site base URL
    base_url = 'https://fantasy.premierleague.com/api/'

    # Fetch a df that contains all finished games per player and a df that contains the upcoming games per player - both of current season
    curr_season_jasons, upcoming_fixt_df = collate_player(base_url)

    # Build current season df
    curr_season_df = build_curr_season(curr_season_jasons, id_name_dict, name_team_dict, name_pos_dict, id_teams_dict)

    # Load pre-trained Gensim Word2Vec model
    w2v_model = api.load('word2vec-google-news-300')

    # Build history df
    history_df = build_history(curr_season_df, last_two_seasons)

    # Build scores df
    scores_df = build_scores_table(curr_season_df, upcoming_fixt_df, name_team_dict, history_df, id_teams_dict, w2v_model)

    # Fetch dfs of upcoming games rival teams and the corresponding difficulty of each game
    team_fdr_df, team_fixt_df = get_fixture_dfs()

    # Build difficulties df
    difficulties_df = build_difficulties_table(team_fdr_df, team_fixt_df, teams_names_dict, scores_df, history_df)

    # Save scores and difficulties dfs
    scores_df.to_csv("scores_df.csv", index=False)
    difficulties_df.to_csv("difficulties_df.csv", index=False)

if __name__ == "__main__":
    get_scores_and_difficulties()

