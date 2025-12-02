import pandas as pd
import numpy as np
import joblib

# Scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load the Dataset
# Ensure 'house_prices_dataset.csv' is in the same folder as this script
print("Loading dataset...")
try:
    df = pd.read_csv('house_prices_dataset.csv') # [cite: 18-20]
except FileNotFoundError:
    print("Error: 'house_prices_dataset.csv' not found.")
    exit()

# 2. Define Features (X) and Target (y)
# [cite: 156-157]
X = df.drop(['price'], axis=1)
y = df['price']

# 3. Create Preprocessing Pipeline
# We use StandardScaler because the PDF mentions handling outliers with Scaling [cite: 125, 159]
numerical_features = ['square_feet', 'num_rooms', 'age', 'distance_to_city(km)'] # [cite: 160]

# Define the transformer
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features)
    ])

# 4. Split Data (80% Train, 20% Test)
# [cite: 170]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Training
print("Training model...")

# Fit the preprocessor on training data
X_train_processed = preprocessor.fit_transform(X_train) # [cite: 174]

# Initialize and train the Decision Tree
model = DecisionTreeRegressor(random_state=42) # [cite: 179]
model.fit(X_train_processed, y_train) # [cite: 181]

# 6. Evaluation (Calculating "Accuracy")
# [cite: 186-193]
print("Evaluating performance...")

# Preprocess the test set (using the scaler fitted on training data)
X_test_processed = preprocessor.transform(X_test)
y_pred = model.predict(X_test_processed)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print results to terminal (matches your request for the screenshot look)
print("-" * 40)
print("HASIL EVALUASI MODEL (DECISION TREE)")
print(f"R-squared (Accuracy): {r2:.4f}")  # Target > 0.81 based on your PDF [cite: 16]
print(f"RMSE Error:           ${rmse:,.2f}")
print("-" * 40)

# 7. Save the Model and Preprocessor
# We need both files for the Streamlit app
joblib.dump(preprocessor, 'preprocessor.sav')
joblib.dump(model, 'house_model.sav')

# Create a dictionary of scores
scores = {'r2': r2, 'rmse': rmse}

# Save it
joblib.dump(scores, 'model_scores.sav')
print("Scores saved to 'model_scores.sav'")

print("Success! Files saved: 'preprocessor.sav' and 'house_model.sav'")