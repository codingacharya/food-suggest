import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------
# Load and Clean Data
# -----------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("final_dataset.csv")

    # ✅ Auto-correct diet type based on recipe name
    non_veg_keywords = ['chicken', 'egg', 'fish', 'mutton', 'prawn', 'meat', 'beef', 'shrimp']
    df['Diet_Type'] = df['Recipe_Name'].apply(
        lambda x: 'Non-Vegetarian' if any(word.lower() in x.lower() for word in non_veg_keywords) else 'Vegetarian'
    )

    # Handle missing values
    df.fillna({
        'Nutrient_Score': df['Nutrient_Score'].mean(),
        'Preference_Score': df['Preference_Score'].mean(),
        'Calories': df['Calories'].mean(),
        'Protein_g': df['Protein_g'].mean(),
        'Carbs_g': df['Carbs_g'].mean(),
        'Fat_g': df['Fat_g'].mean(),
        'User_Rating': df['User_Rating'].mean()
    }, inplace=True)

    return df

df = load_and_clean_data()

st.set_page_config(page_title="AI Nutrition Recommendation System", layout="wide")
st.title("🥗 AI-Powered Personalized Nutrition Recommender")
st.markdown("Get **personalized meal recommendations** based on your health, nutrition, and taste preferences — powered by an ensemble of ML models.")

# -----------------------------------------------------------
# Sidebar Filters
# -----------------------------------------------------------
st.sidebar.header("🔍 Filter Options")

age = st.sidebar.slider("Select Age", 18, 60, (25, 40))
gender = st.sidebar.selectbox("Gender", ["All"] + sorted(df["Gender"].dropna().unique().tolist()))
health = st.sidebar.selectbox("Health Condition", ["All"] + sorted(df["Health_Condition"].dropna().unique().tolist()))
diet = st.sidebar.selectbox("Dietary Restriction", ["All"] + sorted(df["Dietary_Restriction"].dropna().unique().tolist()))
cuisine = st.sidebar.selectbox("Preferred Cuisine", ["All"] + sorted(df["Preferred_Cuisine"].dropna().unique().tolist()))
taste = st.sidebar.selectbox("Taste Preference", ["All"] + sorted(df["Taste_Preference"].dropna().unique().tolist()))

filtered = df[
    (df["Age"].between(age[0], age[1]))
    & ((df["Gender"] == gender) if gender != "All" else True)
    & ((df["Health_Condition"] == health) if health != "All" else True)
    & ((df["Dietary_Restriction"] == diet) if diet != "All" else True)
    & ((df["Preferred_Cuisine"] == cuisine) if cuisine != "All" else True)
    & ((df["Taste_Preference"] == taste) if taste != "All" else True)
]

if filtered.empty:
    st.warning("⚠️ No matches found for selected filters. Try changing some options.")
else:
    st.success(f"✅ Found {len(filtered)} matching records!")

# -----------------------------------------------------------
# ML Ensemble Model (RandomForest + XGBoost + MLP)
# -----------------------------------------------------------
st.subheader("🧠 AI Model Training (Simulated)")

# Encode categorical features
model_df = df.copy()
enc = LabelEncoder()
for col in ['Gender', 'Health_Condition', 'Dietary_Restriction', 'Preferred_Cuisine', 'Taste_Preference', 'Diet_Type']:
    model_df[col] = enc.fit_transform(model_df[col].astype(str))

# Features & Target
X = model_df[['Age', 'BMI', 'Calories', 'Protein_g', 'Carbs_g', 'Fat_g', 'Fiber_g',
              'Vitamin_C_mg', 'Iron_mg', 'User_Rating', 'Nutrient_Score',
              'Preference_Score', 'Gender', 'Health_Condition',
              'Dietary_Restriction', 'Preferred_Cuisine', 'Taste_Preference', 'Diet_Type']]
y = model_df['Recommended']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=80, random_state=42)
xgb = XGBClassifier(n_estimators=80, random_state=42, eval_metric='logloss')
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# Ensemble prediction (average of probabilities)
rf_pred = rf.predict_proba(X_test)[:, 1]
xgb_pred = xgb.predict_proba(X_test)[:, 1]
mlp_pred = mlp.predict_proba(X_test)[:, 1]

ensemble_pred = (rf_pred + xgb_pred + mlp_pred) / 3
ensemble_binary = (ensemble_pred > 0.5).astype(int)
acc = accuracy_score(y_test, ensemble_binary)

st.info(f"✅ Ensemble Model Accuracy: **{acc*100:.2f}%**")

# Apply model on filtered data
filtered_encoded = filtered.copy()
for col in ['Gender', 'Health_Condition', 'Dietary_Restriction', 'Preferred_Cuisine', 'Taste_Preference', 'Diet_Type']:
    filtered_encoded[col] = enc.fit_transform(filtered_encoded[col].astype(str))

filtered_encoded['Predicted_Prob'] = (
    rf.predict_proba(filtered_encoded[X.columns])[:, 1]
    + xgb.predict_proba(filtered_encoded[X.columns])[:, 1]
    + mlp.predict_proba(filtered_encoded[X.columns])[:, 1]
) / 3

filtered_encoded['AI_Score'] = (
    0.6 * filtered_encoded['Nutrient_Score'] +
    0.4 * filtered_encoded['Preference_Score'] +
    filtered_encoded['Predicted_Prob'] * 5
)

top_recommendations = filtered_encoded.sort_values(by="AI_Score", ascending=False).head(10)

# -----------------------------------------------------------
# Display Recommendations
# -----------------------------------------------------------
st.subheader("🍽️ Top Recommended Meals for You")

for _, row in top_recommendations.iterrows():
    with st.container():
        st.markdown(f"### 🥘 {row['Recipe_Name']}  ({row['Cuisine']})")
        col1, col2, col3 = st.columns([2, 2, 1.5])

        with col1:
            st.write(f"**Diet Type:** {row['Diet_Type']}")
            st.write(f"**Cooking Time:** {row['Cooking_Time']} mins")
            st.write(f"**Calories:** {row['Calories']:.1f} kcal")
            st.write(f"**Protein:** {row['Protein_g']} g | **Carbs:** {row['Carbs_g']} g | **Fat:** {row['Fat_g']} g")
            st.write(f"**Fiber:** {row['Fiber_g']} g | **Vitamin C:** {row['Vitamin_C_mg']} mg | **Iron:** {row['Iron_mg']} mg")

        with col2:
            st.metric("⭐ User Rating", f"{row['User_Rating']}/5")
            st.metric("💪 Nutrient Score", f"{row['Nutrient_Score']:.2f}")
            st.metric("🧠 Preference Score", f"{row['Preference_Score']:.2f}")
            st.metric("🤖 AI Score", f"{row['AI_Score']:.2f}")

        with col3:
            if row['Predicted_Prob'] > 0.5:
                st.success("✅ Recommended by AI")
            else:
                st.warning("⚠️ Low Match")

        st.divider()

# -----------------------------------------------------------
# Summary Statistics
# -----------------------------------------------------------
st.subheader("📊 Nutritional Insights")
colA, colB, colC = st.columns(3)
colA.metric("Average Calories", f"{filtered['Calories'].mean():.1f} kcal")
colB.metric("Average Protein", f"{filtered['Protein_g'].mean():.1f} g")
colC.metric("Average Fiber", f"{filtered['Fiber_g'].mean():.1f} g")

st.markdown("---")
st.caption("© 2025 AI Nutrition Recommendation System | Powered by Streamlit, XGBoost, and Scikit-learn")
