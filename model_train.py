# model_train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Generate a robust synthetic dataset resembling the Kaggle ASD Screening Data
np.random.seed(42)
num_samples = 1000

# Generating scores for 10 behavioral questions (0 or 1)
behavioral_scores = np.random.randint(0, 2, size=(num_samples, 10))
age = np.random.randint(1, 40, size=(num_samples, 1))
jaundice = np.random.randint(0, 2, size=(num_samples, 1))  # 0: No, 1: Yes
gender = np.random.randint(0, 2, size=(num_samples, 1))    # 0: Female, 1: Male

# Combine into a DataFrame
columns = [f'A{i}' for i in range(1, 11)] + ['age', 'jaundice', 'gender']
X = np.hstack((behavioral_scores, age, jaundice, gender))
df = pd.DataFrame(X, columns=columns)

# Define a logic-based target variable (ASD traits correlation)
# In real datasets, higher cumulative behavioral scores indicate higher ASD likelihood
total_score = df[[f'A{i}' for i in range(1, 11)]].sum(axis=1) + (df['jaundice'] * 1.5)
df['asd_traits'] = np.where(total_score >= 6, 1, 0)

# 2. Split Features and Target
X_data = df.drop('asd_traits', axis=1)
y_data = df['asd_traits']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
# 3. Train the Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy for documentation
accuracy = model.score(X_test, y_test)
print(f"Model successfully trained with an Accuracy of: {accuracy * 100:.2f}%")

# 4. Save the model configuration
with open('autism_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved as 'autism_model.pkl'")