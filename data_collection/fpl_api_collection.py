import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from functools import partial

base_url = 'https://fantasy.premierleague.com/api/'


def get_bootstrap_data():
    resp = requests.get(f'{base_url}bootstrap-static/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    else:
        return resp.json()


def get_fixture_data():
    resp = requests.get(f'{base_url}fixtures/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    else:
        return resp.json()


def get_player_data(player_id):
    resp = requests.get(f'{base_url}element-summary/{player_id}/')
    if resp.status_code != 200:
        raise Exception(f'Response was status code {resp.status_code}')
    else:
        return resp.json()

def remove_moved_players(df):
    strings = ['loan', 'Loan', 'Contract cancelled', 'Left the club',
               'Permanent', 'Released', 'Signed for', 'Transferred',
               'Season long', 'Not training', 'permanent', 'transferred']
    df_copy = df.loc[~df['news'].str.contains('|'.join(strings), case=False)]
    return df_copy

def get_player_id_dict(web_name=True):
    elements_df = pd.DataFrame(get_bootstrap_data()['elements'])
    elements_df = remove_moved_players(elements_df)
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    elements_df['team_name'] = elements_df['team'].map(teams_df.set_index('id')['short_name'])
    if web_name == True:
        id_dict = dict(zip(elements_df['id'], elements_df['web_name']))
    else:
        elements_df['full_name'] = elements_df['first_name'] + ' ' + \
            elements_df['second_name'] + ' (' + elements_df['team_name'] + ')'
        id_dict = dict(zip(elements_df['id'], elements_df['full_name']))
    return id_dict

def get_name_to_dicts():
    elements_df = pd.DataFrame(get_bootstrap_data()['elements'])
    elements_df = remove_moved_players(elements_df)
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    elements_df['team_name'] = elements_df['team'].map(teams_df.set_index('id')['name'])
    elements_df['full_name'] = elements_df['first_name'] + ' ' + elements_df['second_name']
    name_dict = dict(zip(elements_df['full_name'], elements_df['id']))
    name_to_team_dict = dict(zip(elements_df['full_name'], elements_df['team_name']))
    return name_dict, name_to_team_dict

def get_name_and_pos_and_team_dict(web_name=True):
    elements_df = pd.DataFrame(get_bootstrap_data()['elements'])
    elements_df = remove_moved_players(elements_df)
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    if web_name:
        elements_df['team_name'] = elements_df['team'].map(teams_df.set_index('id')['name'])
        name_to_pos_dict = dict(zip(elements_df['web_name'], elements_df['element_type']))
        name_to_team_dict = dict(zip(elements_df['web_name'], elements_df['team_name']))
    else:
        elements_df['team_name'] = elements_df['team'].map(teams_df.set_index('id')['name'])
        elements_df['full_name'] = elements_df['first_name'] + ' ' + elements_df['second_name']
        name_to_pos_dict = dict(zip(elements_df['full_name'], elements_df['element_type']))
        name_to_team_dict = dict(zip(elements_df['full_name'], elements_df['team_name']))
    return name_to_pos_dict, name_to_team_dict


def fetch_player_data(p_id, p_name, base_url):
    resp = requests.get(f'{base_url}element-summary/{p_id}/')
    if resp.status_code != 200:
        print(f'Request to {p_name} data failed')
        raise Exception(f'Response was status code {resp.status_code}')
    else:
        data = resp.json()
        history = data["history"]
        fixtures = data["fixtures"]
        for fixt_dict in fixtures:
            fixt_dict["p_name"] = p_name
            fixt_dict["p_id"] = p_id
        return history, fixtures

def collate_player(base_url):
    p_dict = get_player_id_dict()
    with ThreadPoolExecutor() as executor:
        # Fetch player data in parallel
        fetch_func = partial(fetch_player_data, base_url=base_url)
        results = executor.map(fetch_func, p_dict.keys(), p_dict.values())
    
    res_hist, res_fixt = zip(*results)
    # Concatenate dataframes
    hist_df = pd.DataFrame(res_hist)
    fixt_df = pd.concat([pd.DataFrame(fixts) for fixts in res_fixt], ignore_index=True)
    return hist_df, fixt_df


def get_league_table():
    fixt_df = pd.DataFrame(get_fixture_data())
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    teams_id_list = teams_df['id'].unique().tolist()
    df_list = []
    for t_id in teams_id_list:
        home_data = fixt_df.copy().loc[fixt_df['team_h'] == t_id]
        away_data = fixt_df.copy().loc[fixt_df['team_a'] == t_id]
        home_data.loc[:, 'was_home'] = True
        away_data.loc[:, 'was_home'] = False
        df = pd.concat([home_data, away_data])
        # df = df.loc[df['finished'] == True]
        df.sort_values('event', inplace=True)
        df.loc[(df['was_home'] == True) &
               (df['team_h_score'] > df['team_a_score']), 'win'] = True
        df.loc[(df['was_home'] == False) &
               (df['team_a_score'] > df['team_h_score']), 'win'] = True
        df.loc[(df['team_h_score'] == df['team_a_score']), 'draw'] = True
        df.loc[(df['was_home'] == True) &
               (df['team_h_score'] < df['team_a_score']), 'loss'] = True
        df.loc[(df['was_home'] == False) &
               (df['team_a_score'] < df['team_h_score']), 'loss'] = True
        df.loc[(df['was_home'] == True), 'gf'] = df['team_h_score']
        df.loc[(df['was_home'] == False), 'gf'] = df['team_a_score']
        df.loc[(df['was_home'] == True), 'ga'] = df['team_a_score']
        df.loc[(df['was_home'] == False), 'ga'] = df['team_h_score']
        df.loc[(df['win'] == True), 'result'] = 'W'
        df.loc[(df['draw'] == True), 'result'] = 'D'
        df.loc[(df['loss'] == True), 'result'] = 'L'
        df.loc[(df['was_home'] == True) &
               (df['team_a_score'] == 0), 'clean_sheet'] = True
        df.loc[(df['was_home'] == False) &
               (df['team_h_score'] == 0), 'clean_sheet'] = True
        ws = len(df.loc[df['win'] == True])
        ds = len(df.loc[df['draw'] == True])
        finished_df = df.loc[df['finished'] == True]
        l_data = {'id': [t_id], 'GP': [len(finished_df)], 'W': [ws], 'D': [ds],
                  'L': [len(df.loc[df['loss'] == True])],
                  'GF': [df['gf'].sum()], 'GA': [df['ga'].sum()],
                  'GD': [df['gf'].sum() - df['ga'].sum()],
                  'CS': [df['clean_sheet'].sum()], 'Pts': [(ws*3) + ds],
                  'Form': [finished_df['result'].tail(5).str.cat(sep='')]}
        df_list.append(pd.DataFrame(l_data))
    league_df = pd.concat(df_list)
    league_df['team'] = league_df['id'].map(teams_df.set_index('id')['short_name'])
    league_df.drop('id', axis=1, inplace=True)
    league_df.reset_index(drop=True, inplace=True)
    league_df.loc[league_df['team'] == 'EVE', 'Pts'] = league_df['Pts'] - 10
    league_df.sort_values(['Pts', 'GD', 'GF', 'GA'], ascending=False, inplace=True)
    league_df.set_index('team', inplace=True)
    league_df['GF'] = league_df['GF'].astype(int)
    league_df['GA'] = league_df['GA'].astype(int)
    league_df['GD'] = league_df['GD'].astype(int)

    league_df['Pts/Game'] = (league_df['Pts']/league_df['GP']).round(2)
    league_df['GF/Game'] = (league_df['GF']/league_df['GP']).round(2)
    league_df['GA/Game'] = (league_df['GA']/league_df['GP']).round(2)
    league_df['CS/Game'] = (league_df['CS']/league_df['GP']).round(2)
    
    return league_df


def get_current_gw():
    events_df = pd.DataFrame(get_bootstrap_data()['events'])
    current_gw = events_df.loc[events_df['is_next'] == True].reset_index()['id'][0]
    return current_gw


def get_fixture_dfs():
    fixt_df = pd.DataFrame(get_fixture_data())
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    teams_list = teams_df['short_name'].unique().tolist()
    fixt_df['team_h'] = fixt_df['team_h'].map(teams_df.set_index('id')['short_name'])
    fixt_df['team_a'] = fixt_df['team_a'].map(teams_df.set_index('id')['short_name'])
    gw_dict = dict(zip(range(1,381),
                       [num for num in range(1, 39) for x in range(10)]))
    fixt_df['event_lock'] = fixt_df['id'].map(gw_dict)
    team_fdr_data = []
    team_fixt_data = []
    for team in teams_list:
        home_data = fixt_df.copy().loc[fixt_df['team_h'] == team]
        away_data = fixt_df.copy().loc[fixt_df['team_a'] == team]
        home_data.loc[:, 'was_home'] = True
        away_data.loc[:, 'was_home'] = False
        df = pd.concat([home_data, away_data])
        df.sort_values('event_lock', inplace=True)
        h_filt = (df['team_h'] == team) & (df['event'].notnull())
        a_filt = (df['team_a'] == team) & (df['event'].notnull())
        df.loc[h_filt, 'next'] = df['team_a'] + ' (H)'
        df.loc[a_filt, 'next'] = df['team_h'] + ' (A)'
        df.loc[df['event'].isnull(), 'next'] = 'BLANK'
        df.loc[h_filt, 'next_fdr'] = df['team_h_difficulty']
        df.loc[a_filt, 'next_fdr'] = df['team_a_difficulty']
        team_fixt_data.append(pd.DataFrame([team] + list(df['next'])).transpose())
        team_fdr_data.append(pd.DataFrame([team] + list(df['next_fdr'])).transpose())
    team_fdr_df = pd.concat(team_fdr_data).set_index(0)
    team_fixt_df = pd.concat(team_fixt_data).set_index(0)
    return team_fdr_df, team_fixt_df


def get_fixtures_for_table():
    fixt_df = pd.DataFrame(get_fixture_data())
    teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
    teams_list = teams_df['short_name'].unique().tolist()
    league_df = get_league_table().reset_index()
    fixt_df['team_h'] = fixt_df['team_h'].map(teams_df.set_index('id')['short_name'])
    fixt_df['team_a'] = fixt_df['team_a'].map(teams_df.set_index('id')['short_name'])
    
    gw_dict = dict(zip(range(1,381),
                       [num for num in range(1, 39) for x in range(10)]))
    fixt_df['event_lock'] = fixt_df['id'].map(gw_dict)
    team_fdr_data = []
    team_fixt_data = []
    team_ga_data = []
    team_gf_data = []
    for team in teams_list:
        home_data = fixt_df.copy().loc[fixt_df['team_h'] == team]
        away_data = fixt_df.copy().loc[fixt_df['team_a'] == team]
        home_data.loc[:, 'was_home'] = True
        away_data.loc[:, 'was_home'] = False
        df = pd.concat([home_data, away_data])
        df.sort_values(['kickoff_time'], inplace=True)
        h_filt = (df['team_h'] == team) & (df['event'].notnull())
        a_filt = (df['team_a'] == team) & (df['event'].notnull())
        df.loc[h_filt, 'next'] = df['team_a'] + ' (H)'
        df.loc[a_filt, 'next'] = df['team_h'] + ' (A)'
        df['team'] = df['next'].str[:3]
        dup_df = df.duplicated(subset=['event'], keep=False).reset_index()
        dup_df.columns = ['index', 'multiple']
        df = df.reset_index().merge(dup_df, on='index', how='left')
        df.set_index('index', inplace=True)
        df.loc[h_filt, 'next_fdr'] = df['team_h_difficulty']
        df.loc[a_filt, 'next_fdr'] = df['team_a_difficulty']
        new_df = df.merge(league_df[['team', 'GA/Game', 'GF/Game']], on='team', how='left')
        event_df = pd.DataFrame({'event': [num for num in range(1, 39)]})
        dedup_df = df.groupby('event').agg({'next': ' + '.join}).reset_index()
        dedup_fdr_df = new_df.groupby('event')[['next_fdr', 'GA/Game', 'GF/Game']].mean().reset_index()
        dedup_df = dedup_df.merge(dedup_fdr_df, on='event', how='left')
        join_df = event_df.merge(dedup_df, on='event', how='left')
        join_df.loc[join_df['next'].isnull(), 'next'] = 'BLANK'
        join_df['GA/Game'] = join_df['GA/Game'].apply(lambda x: round(x, 2))
        join_df['GF/Game'] = join_df['GF/Game'].apply(lambda x: round(x, 2))
        team_fixt_data.append(pd.DataFrame([team] + list(join_df['next'])).transpose())
        team_fdr_data.append(pd.DataFrame([team] + list(join_df['next_fdr'])).transpose())
        team_ga_data.append(pd.DataFrame([team] + list(join_df['GA/Game'])).transpose())
        team_gf_data.append(pd.DataFrame([team] + list(join_df['GF/Game'])).transpose())
    team_fdr_df = pd.concat(team_fdr_data).set_index(0)
    team_fixt_df = pd.concat(team_fixt_data).set_index(0)
    team_ga_df = pd.concat(team_ga_data).set_index(0)
    team_gf_df = pd.concat(team_gf_data).set_index(0)
    return team_fdr_df, team_fixt_df, team_ga_df, team_gf_df
    