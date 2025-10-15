import pandas as pd
import random
import numpy as np

# ===== Step 1: Generate User Profiles =====
num_users = 50

users = pd.DataFrame({
    'User_ID': [f'U{i+1:03}' for i in range(num_users)],
    'Age': [random.randint(18, 60) for _ in range(num_users)],
    'Gender': [random.choice(['Male', 'Female']) for _ in range(num_users)],
    'BMI': [round(random.uniform(18.0, 30.0), 1) for _ in range(num_users)],
    'Health_Condition': [random.choice(['None', 'Iron Deficiency', 'Vitamin D Deficiency', 'Diabetes']) for _ in range(num_users)],
    'Dietary_Restriction': [random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan']) for _ in range(num_users)],
    'Preferred_Cuisine': [random.choice(['Indian', 'Italian', 'Asian', 'Mediterranean']) for _ in range(num_users)],
    'Taste_Preference': [random.choice(['Spicy', 'Sweet', 'Mild', 'Savory']) for _ in range(num_users)]
})
users.to_csv('user_profiles.csv', index=False)
print("✅ user_profiles.csv created successfully!")

# ===== Step 2: Generate Nutrition Database =====
num_foods = 300

nutrition = pd.DataFrame({
    'Food_ID': [f'F{i+1:03}' for i in range(num_foods)],
    'Food_Name': [random.choice(['Apple', 'Chicken', 'Rice', 'Broccoli', 'Spinach', 'Fish', 'Tofu', 'Paneer', 'Oats', 'Banana']) for _ in range(num_foods)],
    'Calories': [random.randint(30, 600) for _ in range(num_foods)],
    'Protein_g': [round(random.uniform(0.1, 35.0), 1) for _ in range(num_foods)],
    'Fat_g': [round(random.uniform(0.1, 25.0), 1) for _ in range(num_foods)],
    'Carbs_g': [round(random.uniform(5.0, 60.0), 1) for _ in range(num_foods)],
    'Fiber_g': [round(random.uniform(0.1, 10.0), 1) for _ in range(num_foods)],
    'Vitamin_C_mg': [round(random.uniform(1.0, 90.0), 1) for _ in range(num_foods)],
    'Iron_mg': [round(random.uniform(0.1, 5.0), 1) for _ in range(num_foods)],
    'Calcium_mg': [round(random.uniform(5.0, 300.0), 1) for _ in range(num_foods)]
})
nutrition.to_csv('nutrition.csv', index=False)
print("✅ nutrition.csv created successfully!")

# ===== Step 3: Generate Recipes =====
num_recipes = 100

recipes = pd.DataFrame({
    'Recipe_ID': [f'R{i+1:03}' for i in range(num_recipes)],
    'Recipe_Name': [random.choice(['Grilled Chicken', 'Veg Stir Fry', 'Paneer Tikka', 'Quinoa Salad', 'Spinach Soup', 'Fish Curry', 'Tofu Bowl', 'Lentil Soup']) for _ in range(num_recipes)],
    'Cuisine': [random.choice(['Indian', 'Italian', 'Asian', 'Mediterranean']) for _ in range(num_recipes)],
    'Diet_Type': [random.choice(['Vegan', 'Vegetarian', 'High-Protein', 'Low-Carb']) for _ in range(num_recipes)],
    'Cooking_Time': [random.randint(10, 45) for _ in range(num_recipes)],
    'Calories': [random.randint(150, 500) for _ in range(num_recipes)],
    'Protein_g': [round(random.uniform(5, 35), 1) for _ in range(num_recipes)],
    'Fat_g': [round(random.uniform(2, 20), 1) for _ in range(num_recipes)],
    'Carbs_g': [round(random.uniform(10, 60), 1) for _ in range(num_recipes)],
    'Fiber_g': [round(random.uniform(1, 10), 1) for _ in range(num_recipes)],
    'Vitamin_C_mg': [round(random.uniform(5, 90), 1) for _ in range(num_recipes)],
    'Iron_mg': [round(random.uniform(0.1, 5.0), 1) for _ in range(num_recipes)],
    'Calcium_mg': [round(random.uniform(10, 200), 1) for _ in range(num_recipes)]
})
recipes.to_csv('recipes.csv', index=False)
print("✅ recipes.csv created successfully!")

# ===== Step 4: Generate User Feedback =====
interactions = []
for _ in range(500):  # random interactions
    user = random.choice(users['User_ID'])
    recipe = random.choice(recipes['Recipe_ID'])
    rating = random.randint(1, 5)
    interactions.append((user, recipe, rating))

feedback = pd.DataFrame(interactions, columns=['User_ID', 'Recipe_ID', 'User_Rating'])
feedback.to_csv('user_feedback.csv', index=False)
print("✅ user_feedback.csv created successfully!")

# ===== Step 5: Merge Everything =====
merged = feedback.merge(users, on='User_ID', how='left').merge(recipes, on='Recipe_ID', how='left')

# Add synthetic scores
merged['Nutrient_Score'] = np.random.uniform(0.6, 1.0, len(merged))
merged['Preference_Score'] = np.random.uniform(0.5, 1.0, len(merged))
merged['Recommended'] = (merged['User_Rating'] > 3).astype(int)

merged.to_csv('final_dataset.csv', index=False)
print("✅ final_dataset.csv created successfully!")
print(f"\n📁 All datasets have been generated in this folder.")
print(f"Total records in final dataset: {len(merged)}")
