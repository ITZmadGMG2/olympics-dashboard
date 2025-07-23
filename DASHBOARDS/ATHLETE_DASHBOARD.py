import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from imblearn.pipeline       import Pipeline as ImbPipeline
from imblearn.over_sampling  import SMOTE
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from category_encoders       import TargetEncoder
from sklearn.ensemble        import RandomForestClassifier
def run_athlete_dashboard():

    # 1) Load data
    @st.cache_data
    def load_data():
        #df = pd.read_csv(r'C:\Users\IMAD SAID\Desktop\OLYMPICS\final_ath_df.csv')
        df=pd.read_csv("final_ath_df.csv")

        return df

    df = load_data()

    # 2) Precompute country-level history
    @st.cache_data
    def build_country_history(df):
        country_hist = (
            df[df['won_medal'] == 1]
            .groupby(['country_name','game_season','game_year'])
            .agg(
                medals_at_games=('won_medal','sum'),
                golds=('medal_code', lambda s: (s==3).sum()),
                silvers=('medal_code', lambda s: (s==2).sum()),
                bronzes=('medal_code', lambda s: (s==1).sum())
            )
            .reset_index()
        )
        return country_hist

    country_hist_df = build_country_history(df)

    # 3) Train Random Forest pipeline
    @st.cache_resource
    def train_pipeline(df):
        TARGET= 'won_medal'
        NUM= [
        'age','appearance_num',
        'cum_gold','cum_silver','cum_bronze','cum_total_medals',
        'medals_last_same_season','gold_count_last','silver_count_last','bronze_count_last',
        'is_debut','Host','is_summer',
        'Athlete_medal_rate','event_size'
        ]
        CAT_TE  = ['discipline_title','event_title','country_3_letter_code']
        FEATURES = NUM  + CAT_TE

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), NUM),
            ('te',  TargetEncoder(), CAT_TE),
        ], remainder='drop')

        pipe = ImbPipeline([
            ('prep', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', RandomForestClassifier(
                class_weight='balanced', n_jobs=-1, random_state=42
            )),
        ])
        X = df[FEATURES]
        y = df[TARGET].astype(int)
        pipe.fit(X, y)
        return pipe, FEATURES

    pipe_rf, FEATURES = train_pipeline(df)

    # 4) Sidebar controls
    st.sidebar.title("Olympic Athlete Medal Probability")
    season = st.sidebar.selectbox('Season', ['Summer','Winter'])
    min_year = 2024 if season=='Summer' else 2026
    year = st.sidebar.number_input('Future Game Year', min_value=min_year, max_value=2050, step=4, value=min_year)
    host = st.sidebar.selectbox('Host Country', sorted(df['country_name'].unique()))

    # 5) UI selectors
    st.title(f"🏅 Athlete Medal Probabilities — {season} {year} @ {host}")
    disciplines = df[df['game_season']==season]['discipline_title'].unique()
    discipline = st.selectbox("Discipline", sorted(disciplines))

    events = df[(df['game_season']==season)&(df['discipline_title']==discipline)]['event_title'].unique()
    event = st.selectbox("Event", sorted(events))

    countries = sorted(df['country_name'].unique())
    country = st.selectbox("Athlete Country", countries)

    sub = df.loc[
        (df['game_season']      == season) &
        (df['discipline_title'] == discipline) &
        (df['country_name']     == country),
    ]

    # Pull the unique athlete list from the sub DataFrame
    athletes = sorted(sub['athlete_full_name_x'].unique().tolist())

    athlete = st.selectbox("Athlete", ["<New Athlete>"] + athletes)

    # 6) Build feature row
    if athlete == "<New Athlete>":
        age = st.number_input("Age", value=25, step=1)
        appearance_num = 1
        cum_gold = cum_silver = cum_bronze = cum_total_medals = 0
        is_debut = 1
        # country lags
        past = country_hist_df[
            (country_hist_df['country_name']==country) &
            (country_hist_df['game_season']==season) &
            (country_hist_df['game_year']<year)
        ].sort_values('game_year')
        if not past.empty:
            last = past.iloc[-1]
            medals_last_same_season = last['medals_at_games']
            gold_count_last = last['golds']
            silver_count_last = last['silvers']
            bronze_count_last = last['bronzes']
        else:
            medals_last_same_season = gold_count_last = silver_count_last = bronze_count_last = 0
    else:
        subA = df[
        (df['athlete_full_name_x'] == athlete) &
        (df['game_season'] == season)
        ]
        # sort by year, then by appearance count
        subA = subA.sort_values(['game_year','appearance_num'], ascending=[True,True])
        # take the last row as the most recent appearance
        rec = subA.iloc[-1]

        # 2) Age at the future Games
        birth_year = rec['athlete_year_birth']
        age = year - birth_year
        appearance_num = rec['appearance_num'] + 1
        # 4) Athlete medal lags: start from their cum_*, then bump if they medaled last time
        cum_gold         = rec['cum_gold']
        cum_silver       = rec['cum_silver']
        cum_bronze       = rec['cum_bronze']
        cum_total_medals = rec['cum_total_medals']

        # If they won a medal in that last record, increment the appropriate counter
        if rec['won_medal'] == 1:
            cum_total_medals += 1
            if rec['medal_code'] == 3:
                cum_gold += 1
            elif rec['medal_code'] == 2:
                cum_silver += 1
            elif rec['medal_code'] == 1:
                cum_bronze += 1
        
        is_debut = 0
        
        # Country lags 
        past = country_hist_df[
            (country_hist_df['country_name']==country) &
            (country_hist_df['game_season']==season) &
            (country_hist_df['game_year']< year)
        ].sort_values('game_year')

        if not past.empty:
            last = past.iloc[-1]
            medals_last_same_season = last['medals_at_games']
            gold_count_last         = last['golds']
            silver_count_last       = last['silvers']
            bronze_count_last       = last['bronzes']
        else:
            medals_last_same_season = gold_count_last = silver_count_last = bronze_count_last = 0

    is_summer = int(season=='Summer')
    Host_flag = int(country==host)
    country_code = df[df['country_name']==country]['country_3_letter_code'].iloc[0]
    if appearance_num > 1:
        Athlete_medal_rate = cum_total_medals / (appearance_num - 1)
    else:
        Athlete_medal_rate = 0.0

    # Find most recent past record for this exact event setup
    event_subset = df[
        (df['game_season'] == season) &
        (df['game_year']   < year) &
        (df['discipline_title'] == discipline) &
        (df['event_title'] == event)
    ]

    if not event_subset.empty:
        latest_row = event_subset.sort_values('game_year').iloc[-1]
        event_size = latest_row['event_size']
    else:
        event_size = 0

    row = pd.DataFrame([{
        'age': age,
        'appearance_num': appearance_num,
        'cum_gold': cum_gold,
        'cum_silver': cum_silver,
        'cum_bronze': cum_bronze,
        'cum_total_medals': cum_total_medals,
        'is_debut': is_debut,
        'Host': Host_flag,
        'is_summer': is_summer,
        'discipline_title': discipline,
        'event_title': event,
        'country_3_letter_code': country_code,
        'medals_last_same_season': medals_last_same_season,
        'gold_count_last': gold_count_last,
        'silver_count_last': silver_count_last,
        'bronze_count_last': bronze_count_last,
        'Athlete_medal_rate': Athlete_medal_rate,
        'event_size': event_size,
    }])

    # 7) Predict & display
    probs = pipe_rf.predict_proba(row)[0]
    result = pd.Series(probs, index=['No Medal','Medal'])
    st.subheader("🏅Predicting Medal Probabilities")
    #st.bar_chart(result)

    #altair bar chart
    # build the small DataFrame in the right order
    df_chart = pd.DataFrame({
        'medal': ['No Medal', 'Medal'],
        'prob' : result.reindex(['No Medal', 'Medal']).values
    })


    color_scale = alt.Scale(
        domain=['No Medal','Medal'],
        range = ['#76b5f0','#ffd700']
    )


    base = alt.Chart(df_chart).encode(
        x=alt.X('medal:N', sort=['No Medal','Medal'], title=None),
        y=alt.Y('prob:Q', title='Probability')
    )


    bars = base.mark_bar().encode(
        color=alt.Color('medal:N', scale=color_scale, legend=None)
    )

    # add text labels (formatted as percentages)
    labels = base.mark_text(
        dy=-5,            # shift text up a bit
        color='green'
    ).encode(
        text=alt.Text('prob:Q', format='.1%')
    )

    chart = (bars + labels).properties(
        width=400, height=300, title='Predicted Medal Probabilities'
    )


    st.subheader(f"Medal Probability for {athlete} from {country}")
    st.altair_chart(chart, use_container_width=True)

    # Athlete summary info
    st.markdown("## ATHLETE INFO")
    st.markdown(f"**NAME:** {athlete}    &nbsp;&nbsp;&nbsp; **Country:** {country}")
    st.markdown(f"**AGE:** {age}      &nbsp;&nbsp;&nbsp; **APP:** {appearance_num}")

    # Individual event medal history
    st.markdown("### INDIVIDUAL EVENT MEDAL HISTORY")
    st.markdown(f"🥇 G: {cum_gold}    🥈 S: {cum_silver}    🥉 B: {cum_bronze}")
    st.markdown(f"**Total medals:** {cum_total_medals}")

    #table of values of features used
    st.subheader("Features Used")
    st.dataframe(row)

