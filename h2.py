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
# ✅ Streamlit Config (must be first command)
# -----------------------------------------------------------
st.set_page_config(page_title="AI Nutrition Recommendation System", layout="wide")

# -----------------------------------------------------------
# Load and Clean Data
# -----------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("final_dataset.csv")

    # Auto-correct diet type based on recipe name
    non_veg_keywords = ['chicken', 'egg', 'fish', 'mutton', 'prawn', 'meat', 'beef', 'shrimp']
    if 'Recipe_Name' in df.columns:
        df['Diet_Type'] = df['Recipe_Name'].apply(
            lambda x: 'Non-Vegetarian' if any(word.lower() in str(x).lower() for word in non_veg_keywords) else 'Vegetarian'
        )
    else:
        df['Diet_Type'] = 'Vegetarian'  # fallback if column missing

    # Fill missing numeric columns
    num_cols = ['Nutrient_Score', 'Preference_Score', 'Calories', 'Protein_g',
                'Carbs_g', 'Fat_g', 'User_Rating', 'Fiber_g', 'Vitamin_C_mg', 'Iron_mg']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    return df


df = load_and_clean_data()

# -----------------------------------------------------------
# App Header
# -----------------------------------------------------------
st.title("🥗 AI-Powered Personalized Nutrition Recommender")
st.markdown("Get **personalized meal recommendations** based on your health, nutrition, and taste preferences — powered by an ensemble of ML models.")

# -----------------------------------------------------------
# Sidebar Filters
# -----------------------------------------------------------
st.sidebar.header("🔍 Filter Options")

# Handle missing columns gracefully
def safe_unique(col):
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

age = st.sidebar.slider("Select Age", 18, 60, (25, 40))
gender = st.sidebar.selectbox("Gender", ["All"] + safe_unique("Gender"))
health = st.sidebar.selectbox("Health Condition", ["All"] + safe_unique("Health_Condition"))
diet = st.sidebar.selectbox("Dietary Restriction", ["All"] + safe_unique("Dietary_Restriction"))
cuisine = st.sidebar.selectbox("Preferred Cuisine", ["All"] + safe_unique("Preferred_Cuisine"))
taste = st.sidebar.selectbox("Taste Preference", ["All"] + safe_unique("Taste_Preference"))

# Apply filters safely
filtered = df.copy()
if "Age" in df.columns:
    filtered = filtered[filtered["Age"].between(age[0], age[1])]
if gender != "All" and "Gender" in df.columns:
    filtered = filtered[filtered["Gender"] == gender]
if health != "All" and "Health_Condition" in df.columns:
    filtered = filtered[filtered["Health_Condition"] == health]
if diet != "All" and "Dietary_Restriction" in df.columns:
    filtered = filtered[filtered["Dietary_Restriction"] == diet]
if cuisine != "All" and "Preferred_Cuisine" in df.columns:
    filtered = filtered[filtered["Preferred_Cuisine"] == cuisine]
if taste != "All" and "Taste_Preference" in df.columns:
    filtered = filtered[filtered["Taste_Preference"] == taste]

if filtered.empty:
    st.warning("⚠️ No matches found for selected filters. Try changing some options.")
else:
    st.success(f"✅ Found {len(filtered)} matching records!")

# -----------------------------------------------------------
# ML Ensemble Model (RandomForest + XGBoost + MLP)
# -----------------------------------------------------------
st.subheader("🧠 AI Model Training (Simulated)")

model_df = df.copy()
enc = LabelEncoder()

# Encode safely
for col in ['Gender', 'Health_Condition', 'Dietary_Restriction', 'Preferred_Cuisine', 'Taste_Preference', 'Diet_Type']:
    if col in model_df.columns:
        model_df[col] = enc.fit_transform(model_df[col].astype(str))
    else:
        model_df[col] = 0

# Define features safely
required_features = ['Age', 'BMI', 'Calories', 'Protein_g', 'Carbs_g', 'Fat_g', 'Fiber_g',
                     'Vitamin_C_mg', 'Iron_mg', 'User_Rating', 'Nutrient_Score',
                     'Preference_Score', 'Gender', 'Health_Condition',
                     'Dietary_Restriction', 'Preferred_Cuisine', 'Taste_Preference', 'Diet_Type']

X = model_df[[col for col in required_features if col in model_df.columns]]
y = model_df['Recommended'] if 'Recommended' in model_df.columns else np.random.randint(0, 2, len(model_df))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=80, random_state=42)
xgb = XGBClassifier(n_estimators=80, random_state=42, eval_metric='logloss')
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# Ensemble model accuracy
rf_pred = rf.predict_proba(X_test)[:, 1]
xgb_pred = xgb.predict_proba(X_test)[:, 1]
mlp_pred = mlp.predict_proba(X_test)[:, 1]
ensemble_pred = (rf_pred + xgb_pred + mlp_pred) / 3
ensemble_binary = (ensemble_pred > 0.5).astype(int)
acc = accuracy_score(y_test, ensemble_binary)

st.info(f"✅ Ensemble Model Accuracy: **{acc*100:.2f}%**")

# -----------------------------------------------------------
# Predict on Filtered Data
# -----------------------------------------------------------
if filtered.empty:
    st.warning("⚠️ No matching data available for prediction. Try adjusting filters.")
    st.stop()

filtered_encoded = filtered.copy()
for col in ['Gender', 'Health_Condition', 'Dietary_Restriction', 'Preferred_Cuisine', 'Taste_Preference', 'Diet_Type']:
    if col in filtered_encoded.columns:
        filtered_encoded[col] = enc.fit_transform(filtered_encoded[col].astype(str))
    else:
        filtered_encoded[col] = 0

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
        st.markdown(f"### 🥘 {row.get('Recipe_Name', 'Unknown Recipe')} ({row.get('Cuisine', 'General')})")
        col1, col2, col3 = st.columns([2, 2, 1.5])

        with col1:
            st.write(f"**Diet Type:** {row.get('Diet_Type', 'N/A')}")
            st.write(f"**Cooking Time:** {row.get('Cooking_Time', 'N/A')} mins")
            st.write(f"**Calories:** {row.get('Calories', 0):.1f} kcal")
            st.write(f"**Protein:** {row.get('Protein_g', 0)} g | **Carbs:** {row.get('Carbs_g', 0)} g | **Fat:** {row.get('Fat_g', 0)} g")
            st.write(f"**Fiber:** {row.get('Fiber_g', 0)} g | **Vitamin C:** {row.get('Vitamin_C_mg', 0)} mg | **Iron:** {row.get('Iron_mg', 0)} mg")

        with col2:
            st.metric("⭐ User Rating", f"{row.get('User_Rating', 0)}/5")
            st.metric("💪 Nutrient Score", f"{row.get('Nutrient_Score', 0):.2f}")
            st.metric("🧠 Preference Score", f"{row.get('Preference_Score', 0):.2f}")
            st.metric("🤖 AI Score", f"{row.get('AI_Score', 0):.2f}")

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
