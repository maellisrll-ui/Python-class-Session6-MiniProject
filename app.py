import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Pokemon Type Classifier",layout="wide")

# ----------------------- Cached loaders (only run once per session)-----------------------------

@st.cache_data
def load_data():
    clean_df = pd.read_csv("data/clean_data.csv")
    try:
        raw_df = pd.read_csv("data/raw_data.csv")
    except FileNotFoundError:
        raw_df = None
    return clean_df, raw_df


@st.cache_resource
def load_model_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, scaler, label_encoder


clean_df, raw_df = load_data()
model, scaler, label_encoder = load_model_artifacts()

FEATURES = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]

# ----------------------- Sidebar navigation-----------------------------

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to", 
    ["Overview", "Data", "EDA", "Model Performance", "Live Prediction"],
                           )

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Maellis, Data source: [PokeAPI](https://pokeapi.co/api/v2/pokemon?limit=200)")
st.sidebar.caption("Last updated: see repository commit history")


# ----------------------- 1. Header -----------------------------
if section == "Overview":
    st.title("Pokemon Type Classifier")
    st.markdown(
        """
        This dashboard predicts the type of a Pokemon based on its features (HP, Attack, Defense, ...). 
        It uses the data pulled for free from [PokeAPI](https://pokeapi.co/api/v2/pokemon?limit=200) - no authentification required. The dataset contains information about 200 Pokemons.
        Use the sidebar to explore the data, visualize the data, to evaluate the model's performance, and to make live predictions.
        """
    )
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Pokemon in dataset", len(clean_df))
    col2.metric("Number of classes", clean_df["type_clean"].nunique())
    col3.metric("Best model", "Logistics Regression")



# ----------------------- 2. Data Overview -----------------------------
elif section == "Data":
    st.title("Data Overview")

    st.write("Compare the raw dataset with the cleaned dataset.")

    tab1, tab2 = st.tabs(["Clean Dataset", "Raw Dataset"])

    with tab1:
        st.subheader("Clean Dataset")
        display_columns = ["name", "hp", "attack", "defense", "special-attack", "special-defense", "speed", "type_clean"]
        st.dataframe(clean_df[display_columns])

    with tab2:
        st.subheader("Raw Dataset")

        if raw_df is not None:
            st.dataframe(raw_df)
        else:
         st.warning("Raw dataset not available.")

    st.markdown("---")
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Clean Dataset")

        st.write(f"Rows: {clean_df.shape[0]}")
        st.write(f"Columns: {clean_df.shape[1]}")
        st.write(f"Missing values: {clean_df.isna().sum().sum()}")

    with col2:

        if raw_df is not None:

            st.write("### Raw Dataset")

            st.write(f"Rows: {raw_df.shape[0]}")
            st.write(f"Columns: {raw_df.shape[1]}")
            st.write(f"Missing values: {raw_df.isna().sum().sum()}")



#----------------------- 3. EDA -----------------------------
elif section == "EDA":

    st.title("Exploratory Data Analysis")

    st.write(
        "Explore the distribution of features and the relationship between features and Pokémon types."
    )

    selected_types = st.multiselect(
        "Select Pokémon types to visualize",
        sorted(clean_df["type_clean"].unique()),
        default=sorted(clean_df["type_clean"].unique())
    )

    filtered_df = clean_df[
        clean_df["type_clean"].isin(selected_types)
    ]

    type_counts = (
        filtered_df["type_clean"]
        .value_counts()
        .reset_index()
    )

    type_counts.columns = [
        "type_clean",
        "count"
    ]

    fig = px.bar(
        type_counts,
        x="type_clean",
        y="count",
        color="type_clean",
        title="Number of Pokémon per Type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


#----------------------- 4. Model Performance Section -----------------------------
elif section == "Model Performance":
    st.title("Model Performance")

    st.write(
        "Evaluate the performance of the trained model using various metrics and visualizations."
    )

    st.subheader("Performance Metrics")

    performance_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [0.35, 0.288, 0.337, 0.306]
    })
    
    st.dataframe(performance_df, use_container_width=True, hide_index=True)

    st.markdown("---")


    st.subheader("Best Model")
    st.success("Logistic Regression")
    st.write(
        "The best model was selected based on cross-validation performance. It achieved the highest accuracy and F1 score among the evaluated models."
    )

    st.markdown("---")

    
    st.subheader("Feature Importance")

    st.info(
        "Feature importance is not directly available for Logistic Regression. However, we can visualize the coefficients of the model to understand the impact of each feature on the prediction.")


#----------------------- 5. Live Prediction -----------------------------
elif section == "Live Prediction":
    
    st.title("Live Prediction")

    st.write(
        "Enter the features of a Pokémon to predict its type using the trained model."
    )

    with st.form("prediction_form"):
        hp = st.number_input("HP", min_value=1, max_value=255, value=50)
        attack = st.number_input("Attack", min_value=1, max_value=190, value=50)
        defense = st.number_input("Defense", min_value=1, max_value=250, value=50)
        special_attack = st.number_input("Special Attack", min_value=1, max_value=194, value=50)
        special_defense = st.number_input("Special Defense", min_value=1, max_value=250, value=50)
        speed = st.number_input("Speed", min_value=1, max_value=180, value=50)
        height = st.number_input("Height", min_value=1, max_value=200, value=10)
        weight = st.number_input("Weight", min_value=1, max_value=1000, value=100)

        submitted = st.form_submit_button("Predict")

        if submitted:

            sample = pd.DataFrame(
                [[
                    hp,
                    attack,
                    defense,
                    special_attack,
                    special_defense,
                    speed,
                    height,
                    weight
                ]],
                columns=[
                    "hp",
                    "attack",
                    "defense",
                    "special_attack",
                    "special-defense",
                    "speed",
                    "height",
                    "weight"
                ]
            )

            sample_scaled = scaler.transform(sample)
            prediction = model.predict(sample_scaled)
            predicted_type = label_encoder.inverse_transform(prediction)[0]


            st.success(f"The predicted type is: **{predicted_type}**")


            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(sample_scaled)[0]

                proba_df = pd.DataFrame({
                    "Type": label_encoder.classes_,
                    "Probability": probabilities
                })

                proba_df = proba_df.sort_values(
                    by="Probability", 
                    ascending=False
                )

                fig = px.bar(
                    proba_df,
                    x="Type",
                    y="Probability",
                    color="Type",
                    title="Prediction Probabilities"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )
