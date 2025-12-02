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
print("Loading dataset...")
try:
    df = pd.read_csv('house_prices_dataset.csv')
except FileNotFoundError:
    print("Error: 'house_prices_dataset.csv' not found.")
    exit()

# 2. Define Features (X) and Target (y)
X = df.drop(['price'], axis=1)
y = df['price']

# 3. Create Preprocessing Pipeline
numerical_features = ['square_feet', 'num_rooms', 'age', 'distance_to_city(km)']

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Training
print("Training model...")

# Fit the preprocessor on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Initialize and train the Decision Tree
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train_processed, y_train)

# 6. Evaluation (Calculating "Accuracy")
print("Evaluating performance...")

# Preprocess the test set (using the scaler fitted on training data)
X_test_processed = preprocessor.transform(X_test)
y_pred = model.predict(X_test_processed)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("-" * 40)
print("HASIL EVALUASI MODEL (DECISION TREE)")
print(f"R-squared (Accuracy): {r2:.4f}")
print(f"RMSE Error:           ${rmse:,.2f}")
print("-" * 40)

# 7. Save the Model and Preprocessor
joblib.dump(preprocessor, 'preprocessor.sav')
joblib.dump(model, 'house_model.sav')

# Create a dictionary of scores
scores = {'r2': r2, 'rmse': rmse}

# Save it
joblib.dump(scores, 'model_scores.sav')
print("Scores saved to 'model_scores.sav'")

print("Success! Files saved: 'preprocessor.sav' and 'house_model.sav'")