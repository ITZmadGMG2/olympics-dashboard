import streamlit as st
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

def run_country_dashboard():

    # 1. Load historical data with features up to the last completed Games
    hist_df = pd.read_csv("OLYMPIC_COUNTRIES.csv")  

    # 2. Helper to compute future-year features based on history
    def compute_future_features(year, season, df, host_country):
        records = []
        for country in df['country_name'].unique():
            subset = df[(df['country_name']==country) &
                        (df['game_season']==season) &
                        (df['game_year']<year)]
            subset = subset.sort_values('game_year')
            # total medal momentum
            M1 = subset['total_medals'].iloc[-1] if len(subset)>0 else 0
            M2 = subset['total_medals'].iloc[-2] if len(subset)>1 else 0
            G1 = subset['golds'].iloc[-1] if len(subset)>0 else 0
            S1 = subset['silvers'].iloc[-1] if len(subset)>0 else 0
            B1 = subset['bronzes'].iloc[-1] if len(subset)>0 else 0
            avg2 = subset['total_medals'].tail(2).mean() if len(subset)>=2 else M1
            diff = M1 - M2
            if M2>0:
                pct = (M1-M2)/M2
            elif M2==0 and M1>0:
                pct = M1
            else:
                pct = 0.0
            # delegation & events from last known Games
            # delegation & events from last 1-2 Games
            if len(subset) >= 2:
                athletes = subset['num_athletes'].iloc[-2:].mean()
                events   = subset['num_events'].iloc[-2:].mean()
            elif len(subset) == 1:
                athletes = subset['num_athletes'].iloc[-1]
                events   = subset['num_events'].iloc[-1]
            else:
                athletes = 0
                events   = 0

            # flags
            is_summer = int(season=='Summer')
            is_host   = int(country==host_country)
            records.append({
                'country_name':     country,
                'game_year':        year,
                'game_season':      season,
                'is_summer':        is_summer,
                'is_host':          is_host,
                'num_athletes':     athletes,
                'num_events':       events,
                'medal_lag1':       M1,
                'last_gold':       G1,
                'last_silver':     S1,
                'last_bronze':     B1,
                'past2games_medal_avg': avg2,
                'past2medal_%_change': pct,
                'past2_medal_diff':     diff,
            })
        return pd.DataFrame(records)

    # 3. Sidebar controls
    st.sidebar.title('Olympic Medal Prediction')

    season     = st.sidebar.selectbox('Season', ['Summer', 'Winter'])
    min_year   = 2024 if season=='Summer' else 2026
    year       = st.sidebar.number_input('Future Game Year', min_value=min_year, max_value=2050, step=4, value=min_year)
    host       = st.sidebar.selectbox('Host Country', sorted(hist_df['country_name'].unique()))
    new_country = st.sidebar.text_input('Add New Country (optional)')
    banned = st.sidebar.multiselect(
        "Exclude countries (e.g. banned)",
        options=sorted(hist_df['country_name'].unique()),
        default=[]
    )
    # 4. Build prediction DataFrame
    pred_df = compute_future_features(year, season, hist_df, host)

    # 5. Filter out any banned countries
    if banned:
        pred_df = pred_df[~pred_df['country_name'].isin(banned)]

    # 5. Allow editing of delegation & events (including new country)
    pred_df = pred_df.set_index('country_name')
    if new_country:
        # add new country with zeros
        pred_df.loc[new_country] = {
            'game_year': year,
            'game_season': season,
            'is_summer': int(season=='Summer'),
            'is_host': int(new_country==host),
            'num_athletes': 0,
            'num_events': 0,
            'medal_lag1': 0,
            'past2games_medal_avg': 0,
            'past2medal_%_change': 0,
            'past2_medal_diff': 0,
            'last_gold': 0,
            'last_silver': 0,
            'last_bronze': 0
        }
    # data editor
    edited = st.data_editor(
        pred_df[['num_athletes','num_events']].reset_index(),
        num_rows='fixed',
        column_config={
            'country_name': st.column_config.Column('Country', disabled=True),
            'num_athletes': st.column_config.NumberColumn('Delegation Size', min_value=0, step=1),
            'num_events':   st.column_config.NumberColumn('Event Count',    min_value=0, step=1)
        }
    )
    # override
    pred_df.loc[edited['country_name'],'num_athletes'] = edited['num_athletes'].values
    pred_df.loc[edited['country_name'],'num_events']   = edited['num_events'].values
    pred_df = pred_df.reset_index()

    # Train selected model and predict
    model_name = 'Random Forest'
    FEATURES = ['is_summer','is_host','num_athletes','num_events',
                'medal_lag1','past2games_medal_avg','past2medal_%_change',
                'past2_medal_diff','last_gold','last_silver','last_bronze']
    TARGETS  = ['golds','silvers','bronzes']

    # instantiate
    base = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)

    model = MultiOutputRegressor(base)
    model.fit(hist_df[FEATURES], hist_df[TARGETS])
    preds = model.predict(pred_df[FEATURES])

    # Build results table
    results = pred_df[['country_name']].copy()
    results['Gold']   = np.round(preds[:,0]).astype(int)
    results['Silver'] = np.round(preds[:,1]).astype(int)
    results['Bronze'] = np.round(preds[:,2]).astype(int)
    results['Total']  = results[['Gold','Silver','Bronze']].sum(axis=1)
    results = results.sort_values('Total', ascending=False).reset_index(drop=True)

    # 8. Display
    st.title(f"🏅 {model_name} Predictions — {season} {year}")
    st.dataframe(results.style.format({
        'Gold🥇':'{:.0f}','Silver🥈':'{:.0f}','Bronze🥉':'{:.0f}','Total🎖':'{:.0f}'
    }))

    st.subheader('Top 10 Predicted Medals')
    top10 = results.head(10).set_index('country_name')[['Gold','Silver','Bronze']]
    st.bar_chart(top10)
