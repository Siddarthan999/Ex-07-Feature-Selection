# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import ExtraTreesRegressor

# Load the dataset
df = pd.read_csv('CarPrice.csv')

# Drop unnecessary columns
df = df.drop(['car_ID', 'CarName'], axis=1)

# Convert categorical columns into numerical columns
le = LabelEncoder()
df['fueltype'] = le.fit_transform(df['fueltype'])
df['aspiration'] = le.fit_transform(df['aspiration'])
df['doornumber'] = le.fit_transform(df['doornumber'])
df['carbody'] = le.fit_transform(df['carbody'])
df['drivewheel'] = le.fit_transform(df['drivewheel'])
df['enginelocation'] = le.fit_transform(df['enginelocation'])
df['enginetype'] = le.fit_transform(df['enginetype'])
df['cylindernumber'] = le.fit_transform(df['cylindernumber'])
df['fuelsystem'] = le.fit_transform(df['fuelsystem'])

# Split the data into features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Apply feature selection techniques on the training dataset
# Method 1: Univariate Selection
selector = SelectKBest(score_func=f_regression, k=10)
X_train_new = selector.fit_transform(X_train, y_train)
mask = selector.get_support()
selected_features = X_train.columns[mask]

# Method 2: Feature Importance using Extra Trees Regressor
model = ExtraTreesRegressor()
model.fit(X_train, y_train)
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
selected_features = X_train.columns[indices][:10]

# Save the important features with the target variable to a new CSV file
df_new = pd.concat([X_train[selected_features], y_train], axis=1)
df_new.to_csv('CarPrice_new.csv', index=False)
        