import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Dataset Simulate/Load karo
np.random.seed(42)
n_samples = 1500

data = {
    'battery_power': np.random.randint(500, 2000, n_samples),
    'clock_speed': np.round(np.random.uniform(0.5, 3.0, n_samples), 1),
    'int_memory': np.random.randint(2, 64, n_samples),
    'n_cores': np.random.randint(1, 9, n_samples),
    'px_height': np.random.randint(100, 1960, n_samples),
    'px_width': np.random.randint(500, 1998, n_samples),
    'ram': np.random.randint(256, 3998, n_samples)
}

df = pd.DataFrame(data)

# Target Column: Price Range (0: Low, 1: Medium, 2: High, 3: Very High)
score = (df['ram'] * 0.7) + (df['battery_power'] * 0.2) + (df['int_memory'] * 0.1)
df['price_range'] = pd.qcut(score, q=4, labels=[0, 1, 2, 3]).astype(int)

X = df.drop(columns=['price_range'])
y = df['price_range']

# 2. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Model ko Save (.pkl) karo
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model train ho gaya hai aur 'model.pkl' save ho chuka hai.")