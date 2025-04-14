import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Title ---
st.title("🐧 Penguin Species Predictor")

st.write("Adjust the features to predict which penguin species it is.")

st.subheader("📄 View the Dataset")

try:
    df = pd.read_csv("penguins.csv")
    if st.checkbox("Show full penguin dataset"):
        st.write(df)
except Exception as e:
    st.error(f"Error loading dataset: {e}")

# --- Load data ---
df = pd.read_csv("penguins.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical features
le_sex = LabelEncoder()
le_island = LabelEncoder()
df["sex_encoded"] = le_sex.fit_transform(df["sex"])
df["island_encoded"] = le_island.fit_transform(df["island"])

# Define features and labels
X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex_encoded", "island_encoded"]]
y = df["species"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# --- Define default values ---
DEFAULTS = {
    "bill_length_mm": 45.0,
    "bill_depth_mm": 17.0,
    "flipper_length_mm": 200.0,
    "body_mass_g": 4200.0,
    "sex": "MALE",         # Must match exactly one of le_sex.classes_
    "island": "Biscoe"     # Must match exactly one of le_island.classes_
}

# --- Handle reset safely ---
if st.sidebar.button("🔄 Reset Inputs"):
    for key, val in DEFAULTS.items():
        st.session_state[key] = val
    st.session_state["reset_triggered"] = True
    st.rerun()

# --- Prevent rerun loop ---
if st.session_state.get("reset_triggered", False):
    st.session_state["reset_triggered"] = False

# --- Initialize session state if not already done ---
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

bill_length = st.sidebar.slider(
    "Bill Length (mm)",
    float(df["bill_length_mm"].min()),
    float(df["bill_length_mm"].max()),
    st.session_state.get("bill_length_mm", DEFAULTS["bill_length_mm"])
)
st.session_state["bill_length_mm"] = bill_length

bill_depth = st.sidebar.slider(
    "Bill Depth (mm)",
    float(df["bill_depth_mm"].min()),
    float(df["bill_depth_mm"].max()),
    st.session_state.get("bill_depth_mm", DEFAULTS["bill_depth_mm"])
)
st.session_state["bill_depth_mm"] = bill_depth

flipper_length = st.sidebar.slider(
    "Flipper Length (mm)",
    float(df["flipper_length_mm"].min()),
    float(df["flipper_length_mm"].max()),
    st.session_state.get("flipper_length_mm", DEFAULTS["flipper_length_mm"])
)
st.session_state["flipper_length_mm"] = flipper_length

body_mass = st.sidebar.slider(
    "Body Mass (g)",
    float(df["body_mass_g"].min()),
    float(df["body_mass_g"].max()),
    st.session_state.get("body_mass_g", DEFAULTS["body_mass_g"])
)
st.session_state["body_mass_g"] = body_mass

sex_options = list(le_sex.classes_)
current_sex = st.session_state.get("sex", DEFAULTS["sex"])

# If current value isn't valid, fall back to first option
if current_sex not in sex_options:
    current_sex = sex_options[0]

sex_index = sex_options.index(current_sex)

sex = st.sidebar.selectbox("Sex", sex_options, index=sex_index)
st.session_state["sex"] = sex

st.session_state["sex"] = sex

island_options = list(le_island.classes_)
current_island = st.session_state.get("island", DEFAULTS["island"])

if current_island not in island_options:
    current_island = island_options[0]

island_index = island_options.index(current_island)

island = st.sidebar.selectbox("Island", island_options, index=island_index)
st.session_state["island"] = island

# Encode user input
sex_encoded = le_sex.transform([sex])[0]
island_encoded = le_island.transform([island])[0]

# Predict 
input_data = pd.DataFrame([[bill_length, bill_depth, flipper_length, body_mass, sex_encoded, island_encoded]],
                          columns=X.columns)

prediction = model.predict(input_data)[0]
probs = model.predict_proba(input_data)[0]
predicted_class_index = model.classes_.tolist().index(prediction)
probability = probs[predicted_class_index]
st.subheader("Predicted Species:")
st.success(f"🐧 Predicted species: **{prediction}** ({probability:.2%} confidence)")

# probability matrix species
st.subheader("🔢 Prediction Probabilities")
proba_df = pd.DataFrame([probs], columns=model.classes_)
st.write(proba_df.T.rename(columns={0: "Probability"}).style.format({"Probability": "{:.2%}"}))

# Show species info 
st.subheader("About this species")

if prediction == "Adelie":
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/03/Adelie_Penguin.jpg", caption="Adelie Penguin", use_container_width=True)
    st.markdown("""
    **Adelie Penguins** are found on the Antarctic continent and nearby islands.  
    They have a white ring around their eyes and eat mostly krill and small fish.
    """)
elif prediction == "Chinstrap":
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/South_Shetland-2016-Deception_Island–Chinstrap_penguin_%28Pygoscelis_antarctica%29_04.jpg", caption="Chinstrap Penguin", use_container_width=True)
    st.markdown("""
    **Chinstrap Penguins** are named for the black line under their heads.  
    They are one of the most aggressive penguin species and live on rocky islands in the Southern Ocean.
    """)
elif prediction == "Gentoo":
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/00/Brown_Bluff-2016-Tabarin_Peninsula–Gentoo_penguin_%28Pygoscelis_papua%29_03.jpg", caption="Gentoo Penguin", use_container_width=True)
    st.markdown("""
    **Gentoo Penguins** are the fastest swimmers of all penguins.  
    They have bright orange bills and live in colonies on many sub-Antarctic islands.
    """)


