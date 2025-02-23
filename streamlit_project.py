"""
Make sure you have the following installed:
- streamlit
- mplsoccer
- pandas
- numpy
- matplotlib.pyplot
"""
import json
import pandas as pd
import streamlit as st
import numpy as np
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt


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

def plot_shots(df, ax, pitch):
    for x in df.to_dict(orient='records'):
        pitch.scatter(
            x=float(x['location'][0]),
            y=float(x['location'][1]),
            ax=ax,
            s=1000 * x['shot_statsbomb_xg'],
            color='green' if x['shot_outcome'] == 'Goal' else 'white',
            edgecolors='black',
            alpha=1 if x['type'] == 'goal' else .5,
            zorder=2 if x['type'] == 'goal' else 1
        )


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
    st.write(match_result)

# Get the outcome of the selected match
match_result = df.loc[df["match_info"] == match, "outcome"].values[0]


player = st.selectbox("Select a player", df[df['match_info'] == match]['player'].sort_values().unique(), index=None)
filtered_df = filter_data(df, match, player)


pitch = VerticalPitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#f0f0f0', line_color='black', half=True)
fig, ax = pitch.draw(figsize=(10, 10))
plot_shots(filtered_df, ax, pitch)

st.pyplot(fig)

########

st.title("Shot Map Visualization")

# Extract X and Y from 'location' column
filtered_df["X"] = filtered_df["location"].apply(lambda loc: float(loc[0]))
filtered_df["Y"] = filtered_df["location"].apply(lambda loc: float(loc[1]))

# Calculate statistics
total_shots = filtered_df.shape[0]
total_goals = filtered_df[filtered_df['shot_outcome'] == 'Goal'].shape[0]
total_xG = filtered_df["shot_statsbomb_xg"].sum()
xG_per_shot = total_xG / total_shots if total_shots > 0 else 0

# Create second shot map
pitch = VerticalPitch(
    pitch_type='statsbomb',
    half=True,
    pitch_color='#0C0D0E',
    line_color='white',
    linewidth=0.75
)

fig, ax = plt.subplots(figsize=(10, 10))  # Reasonable figure size
fig.patch.set_facecolor('#0C0D0E')

pitch.draw(ax=ax)

# Plot shots
for _, shot in filtered_df.iterrows():
    if not np.isnan(shot["X"]) and not np.isnan(shot["Y"]):  # Ensure valid data
        pitch.scatter(
            shot["X"],
            shot["Y"],
            s=1000 * shot["shot_statsbomb_xg"] if shot["shot_statsbomb_xg"] > 0 else 50,
            color='green' if shot["shot_outcome"] == 'Goal' else 'grey',
            ax=ax,
            alpha=0.7,
            linewidth=0.8,
            edgecolor='white'
        )

st.pyplot(fig)

# Display statistics
st.markdown(f"""
    <h3 style='text-align: center; color: white;'>
    Shots: {total_shots} | Goals: {total_goals} | xG: {total_xG:.2f} | xG/Shot: {xG_per_shot:.2f}
    </h3>
""", unsafe_allow_html=True)