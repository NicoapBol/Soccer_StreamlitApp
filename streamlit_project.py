"""
Make sure you have the following installed:
- streamlit
- mplsoccer
- pandas
"""
import json
import pandas as pd
import streamlit as st
from mplsoccer import VerticalPitch


st.title("Argentina World Cup 2022 Analysis")

#picture at the center
image_path = "me.jpg"
st.image(image_path,use_container_width=False, caption="By:Nicolas Acha", output_format="PNG")


st.subheader("Filter a match to see Argentina shots")

df = pd.read_csv('arg_eventsWC22.csv')
df = df[df['type'] == 'Shot'].reset_index(drop=True)
df['location'] = df['location'].apply(json.loads)

def filter_data(df: pd.DataFrame, match: str, player: str):
    if match:
        df = df[df['match_info'] == match]
    if player:
        df = df[df['player'] == player]
    return df

#def plot_shots(df, ax, pitch):
 #   for x in df.to_dict(orient='records'):
  #      pitch.scatter(
   ###       ax=ax,
      #      s=1000 * x['shot_statsbomb_xg'],
       #     color='green' if x['shot_outcome'] == 'Goal' else 'gray',
        #    edgecolors='black',
         #   alpha=1 if x['type'] == 'goal' else .5,
          #  zorder=2 if x['type'] == 'goal' else 1
        #)

##
def plot_shots(df, ax, pitch):
    for x in df.to_dict(orient='records'):
        shot_x = float(x['location'][0])
        shot_y = float(x['location'][1])
        shot_xG = x['shot_statsbomb_xg']

        # ✅ Plot shot location
        pitch.scatter(
            x=shot_x,
            y=shot_y,
            ax=ax,
            s=(shot_xG ** 0.5) * 700 ,
            color='green' if x['shot_outcome'] == 'Goal' else 'gray',
            edgecolors='black',
            alpha=1 if x['shot_outcome'] == 'Goal' else .5,
            zorder=2 if x['shot_outcome'] == 'Goal' else 1,
            marker="^" 
        )

        # ✅ Attach xG text **inside pitch.scatter()** using pitch.annotate()
        pitch.annotate(
            f"{shot_xG:.2f}",
            xy=(shot_x- 2.2, shot_y),
            ax=ax,
            fontsize=10,
            color="#d32d44",
            va="bottom",  # Align text below the marker
            ha="center",
        )


###

# filter the dataframe
#match = st.selectbox("Select a match", df['match_info'].sort_values().unique(), index=None)

# Default match selection
default_match = "Argentina Vs. Saudi Arabia (2022-11-22)"

# Get the list of unique matches
match_options = df["match_info"].sort_values().unique()

# Ensure the default match exists in the list
default_index = 0  # Default to first match in case the desired default is missing
if default_match in match_options:
    default_index = list(match_options).index(default_match)

# Create selectbox with default index
match = st.selectbox("Select a match", match_options, index=default_index)

# Filter dataframe based on match selection
filtered_df = df[df["match_info"] == match]


# Handle missing data safely
if filtered_df.empty:
    st.error("No data found for the selected match.")
elif "outcome" not in filtered_df.columns:
    st.error("The column 'outcome' does not exist in the dataset.")
else:
    match_result = filtered_df["outcome"].iloc[0]
    st.subheader("Match Result")
    st.markdown(f"<h3 style='text-align: center; font-weight: bold;'>{match_result}</h3>", unsafe_allow_html=True)

# Get goal scorers for the selected match
goal_scorers = filtered_df[filtered_df["shot_outcome"] == "Goal"]["player"].unique()

# Display goal scorers if any
if len(goal_scorers) > 0:
    st.markdown(f"<h6 style='text-align: left; font-weight: bold;'>{', '.join(goal_scorers)}</h6>", unsafe_allow_html=True)
else:
    st.markdown(f"<h6 style='text-align: left; font-weight: bold; color: gray;'>No goals scored</h6>", unsafe_allow_html=True)


# Get the outcome of the selected match
match_result = df.loc[df["match_info"] == match, "outcome"].values[0]


player = st.selectbox("Select a player", df[df['match_info'] == match]['player'].sort_values().unique(), index=None)
filtered_df = filter_data(df, match, player)

st.subheader("Shot Map Visualization")

pitch = VerticalPitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#f0f0f0', line_color='black', half=True)
fig, ax = pitch.draw(figsize=(12, 12))
plot_shots(filtered_df, ax, pitch)

st.pyplot(fig)

########


# Calculate statistics
total_shots = filtered_df.shape[0]
total_goals = filtered_df[filtered_df['shot_outcome'] == 'Goal'].shape[0]
total_penalty_goals = filtered_df[(filtered_df["penalty"] == "yes") & (filtered_df["shot_outcome"] == "Goal")].shape[0]
total_xG = filtered_df["shot_statsbomb_xg"].sum()
xG_per_shot = total_xG / total_shots if total_shots > 0 else 0

# Display statistics
st.markdown(f"""
    <h4 style='text-align: center; color: white;'>
    Shots: {total_shots} | Goals: {total_goals} | Penalty Goals: {total_penalty_goals} | xG: {total_xG:.2f} | xG/Shot: {xG_per_shot:.2f}
    </h4>
""", unsafe_allow_html=True)


#######