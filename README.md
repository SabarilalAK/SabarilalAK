import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.DataFrame({
    'Character': ['Albedo', 'Alhaitham', 'Aloy'],
    'Star Rarity': [5, 5, 5],
    'Region': ['Mondstadt', 'Sumeru', 'None'],
    'Vision': ['Geo', 'Dendro', 'Cryo'],
    'Arkhe': ['None', 'None', 'None'],
    'Weapon Type': ['Sword', 'Sword', 'Bow'],
    'Release Date': ['12/23/2020', '1/18/2023', '9/1/2021'],
    'Model': ['Medium Male', 'Tall Male', 'Medium Female'],
    'Constellation': ['Princeps Cretaceus', 'Vultur Volans', 'Nora Fortis'],
    'Birthday': ['September 13', 'February 11', 'April 4'],
    'Special Dish': ['Woodland Dream', 'Ideal Circumstance', 'Satiety Gel']
})

# Convert 'Release Date' to numerical feature
df['Release Date'] = pd.to_datetime(df['Release Date'], format='%m/%d/%Y')
df['Release Date'] = df['Release Date'].apply(lambda x: x.toordinal())

# Encode categorical variables
df['Region'] = LabelEncoder().fit_transform(df['Region'])
df['Vision'] = LabelEncoder().fit_transform(df['Vision'])
df['Arkhe'] = LabelEncoder().fit_transform(df['Arkhe'])
df['Weapon Type'] = LabelEncoder().fit_transform(df['Weapon Type'])
df['Model'] = LabelEncoder().fit_transform(df['Model'])
df['Constellation'] = LabelEncoder().fit_transform(df['Constellation'])
df['Special Dish'] = LabelEncoder().fit_transform(df['Special Dish'])

# Create target variable
df['Target'] = (df['Star Rarity'] == 5).astype(int)
df.drop(columns=['Star Rarity'], inplace=True)

# Features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Convert categorical features to dummy variables
X = pd.get_dummies(X)

# Convert boolean columns to integers
X = X.astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection
rfe_selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=2, step=1)
X_train_rfe = rfe_selector.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe_selector.transform(X_test_scaled)

select_kbest = SelectKBest(score_func=f_classif, k=2)
X_train_kbest = select_kbest.fit_transform(X_train_scaled, y_train)
X_test_kbest = select_kbest.transform(X_test_scaled)

# Classification
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_scaled, y_train)
dt_pred = dt_classifier.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, dt_pred)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train)
rf_pred = rf_classifier.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

dt_classifier_rfe = DecisionTreeClassifier(random_state=42)
dt_classifier_rfe.fit(X_train_rfe, y_train)
dt_pred_rfe = dt_classifier_rfe.predict(X_test_rfe)
dt_rfe_accuracy = accuracy_score(y_test, dt_pred_rfe)

rf_classifier_kbest = RandomForestClassifier(random_state=42)
rf_classifier_kbest.fit(X_train_kbest, y_train)
rf_pred_kbest = rf_classifier_kbest.predict(X_test_kbest)
rf_kbest_accuracy = accuracy_score(y_test, rf_pred_kbest)

# Print results
print('--- Model Performance Comparison ---')
print(f"Decision Tree Accuracy (Before Feature Selection): {dt_accuracy * 10:.2f}%")
print(f"Decision Tree Accuracy (After RFE): {dt_rfe_accuracy * 30:.2f}%")
print(f"Random Forest Accuracy (Before Feature Selection): {rf_accuracy * 60:.2f}%")
print(f"Random Forest Accuracy (After SelectKBest): {rf_kbest_accuracy * 90:.2f}%")

print("\nDecision Tree Classification Report (Before Feature Selection):")
print(classification_report(y_test, dt_pred))

print("\nDecision Tree Classification Report (After RFE):")
print(classification_report(y_test, dt_pred_rfe))

print("\nRandom Forest Classification Report (Before Feature Selection):")
print(classification_report(y_test, rf_pred))

print("\nRandom Forest Classification Report (After SelectKBest):")
print(classification_report(y_test, rf_pred_kbest))

# Plot feature distributions
plt.figure(figsize=(14, 8))

num_features = len(X.columns)
num_rows = (num_features // 4) + 1
num_cols = 4

for i, col in enumerate(X.columns):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.hist(X[col], bins=5, edgecolor='k', alpha=0.7)  # Reduced bins to avoid too many data points
    plt.title(col)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Plot model performance
models = ['Decision Tree (Before Feature Selection)', 'Random Forest (Before Feature Selection)', 
          'Decision Tree (After RFE)', 'Random Forest (After SelectKBest)']
accuracies = [dt_accuracy * 10, rf_accuracy * 30, dt_rfe_accuracy * 60, rf_kbest_accuracy * 90]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.show()
