import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ----------------------------
# API KEYS
# ----------------------------
SPOONACULAR_API_KEY = "YOUR_SPOONACULAR_KEY"
YOUTUBE_API_KEY = "YOUR_YOUTUBE_KEY"

# ----------------------------
# Load and Clean Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_dataset.csv")

    # Fix Diet_Type based on recipe name
    non_veg_keywords = ['chicken', 'egg', 'fish', 'mutton', 'prawn', 'meat', 'beef', 'shrimp']
    df['Diet_Type'] = df['Recipe_Name'].apply(
        lambda x: 'Non-Vegetarian' if any(word.lower() in x.lower() for word in non_veg_keywords) else 'Vegetarian'
    )

    # Fill missing categorical values
    for col in ['Health_Condition','Gender','Dietary_Restriction','Preferred_Cuisine','Taste_Preference']:
        df[col] = df[col].fillna('None').astype(str)

    # Convert numeric columns safely
    numeric_cols = ['Nutrient_Score','Preference_Score','Calories','Protein_g','Carbs_g','Fat_g','Fiber_g','Vitamin_C_mg','Iron_mg','User_Rating']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())

    return df

df = load_data()

st.set_page_config(page_title="AI Nutrition Recommender", layout="wide")
st.title("🥗 AI-Powered Personalized Food Recommendation")

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("Filter Options")
age = st.sidebar.slider("Age", 18, 60, (25, 40))
gender = st.sidebar.selectbox("Gender", ["All"] + sorted(df["Gender"].unique()))
health = st.sidebar.selectbox("Health Condition", ["All"] + sorted(df["Health_Condition"].unique()))
diet = st.sidebar.selectbox("Dietary Restriction", ["All"] + sorted(df["Dietary_Restriction"].unique()))
cuisine = st.sidebar.selectbox("Preferred Cuisine", ["All"] + sorted(df["Preferred_Cuisine"].unique()))
taste = st.sidebar.selectbox("Taste Preference", ["All"] + sorted(df["Taste_Preference"].unique()))

# ----------------------------
# Filter Dataset
# ----------------------------
filtered = df[
    (df["Age"].between(age[0], age[1]))
    & ((df["Gender"] == gender) if gender != "All" else True)
    & ((df["Health_Condition"] == health) if health != "All" else True)
    & ((df["Dietary_Restriction"] == diet) if diet != "All" else True)
    & ((df["Preferred_Cuisine"] == cuisine) if cuisine != "All" else True)
    & ((df["Taste_Preference"] == taste) if taste != "All" else True)
]

if filtered.empty:
    st.warning("⚠️ No matches found for selected filters.")
    print("⚠️ No matches found in console!")
else:
    num_matches = len(filtered)
    st.success(f"✅ Found {num_matches} matching records!")
    print(f"✅ Found {num_matches} matching records in console!")

# ----------------------------
# ML Ensemble Model
# ----------------------------
st.subheader("🧠 AI Model Training (Ensemble)")

# Encode categorical features
model_df = df.copy()
enc = LabelEncoder()
for col in ['Gender','Health_Condition','Dietary_Restriction','Preferred_Cuisine','Taste_Preference','Diet_Type']:
    model_df[col] = enc.fit_transform(model_df[col].astype(str))

X = model_df[['Age','BMI','Calories','Protein_g','Carbs_g','Fat_g','Fiber_g',
              'Vitamin_C_mg','Iron_mg','User_Rating','Nutrient_Score','Preference_Score',
              'Gender','Health_Condition','Dietary_Restriction','Preferred_Cuisine','Taste_Preference','Diet_Type']]
y = model_df['Recommended']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=80, random_state=42)
xgb = XGBClassifier(n_estimators=80, random_state=42, eval_metric='logloss')
mlp = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=300, random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# Ensemble predictions
rf_pred = rf.predict_proba(X_test)[:,1]
xgb_pred = xgb.predict_proba(X_test)[:,1]
mlp_pred = mlp.predict_proba(X_test)[:,1]

ensemble_pred = (rf_pred + xgb_pred + mlp_pred)/3
ensemble_binary = (ensemble_pred>0.5).astype(int)
acc = accuracy_score(y_test, ensemble_binary)
st.info(f"Ensemble Model Accuracy: {acc*100:.2f}%")
print(f"Ensemble Model Accuracy: {acc*100:.2f}%")

# ----------------------------
# Helper Functions for APIs
# ----------------------------
def fetch_spoonacular(recipe_name):
    url = f"https://api.spoonacular.com/recipes/complexSearch?query={recipe_name}&number=1&addRecipeInformation=true&apiKey={SPOONACULAR_API_KEY}"
    res = requests.get(url).json()
    if res.get("results"):
        r = res["results"][0]
        return {"title": r["title"], "image": r.get("image",""), "sourceUrl": r.get("sourceUrl","")}
    return None

def fetch_youtube(recipe_name):
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={recipe_name} recipe&type=video&key={YOUTUBE_API_KEY}&maxResults=1"
    res = requests.get(url).json()
    if res.get("items"):
        video_id = res["items"][0]["id"]["videoId"]
        return f"https://www.youtube.com/watch?v={video_id}"
    return None

# ----------------------------
# Display Top Recommendations
# ----------------------------
st.subheader("🍽️ Top AI Recommendations with Real Recipes & Videos")

top_recipes = filtered.head(10)

for _, row in top_recipes.iterrows():
    with st.container():
        st.markdown(f"### 🥘 {row['Recipe_Name']} ({row['Cuisine']})")
        
        # Spoonacular
        recipe_info = fetch_spoonacular(row['Recipe_Name'])
        if recipe_info:
            if recipe_info['image']:
                st.image(recipe_info['image'], width=300)
            st.markdown(f"[View Full Recipe]({recipe_info['sourceUrl']})")

        # YouTube
        video_url = fetch_youtube(row['Recipe_Name'])
        if video_url:
            st.video(video_url)

        st.write(f"**Diet Type:** {row['Diet_Type']} | **Calories:** {row['Calories']}")
        st.write(f"Protein: {row['Protein_g']}g | Carbs: {row['Carbs_g']}g | Fat: {row['Fat_g']}g")
        st.divider()

st.caption("© 2025 AI Nutrition Recommendation System | Powered by Streamlit, XGBoost, and Scikit-learn")
