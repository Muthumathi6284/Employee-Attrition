import pandas as pd
# Import CSV File

data=pd.read_csv("C:/Users/Hp/Downloads/Employee-Attrition - Employee-Attrition.csv")
df=pd.DataFrame(data)

# Drop Unwanted Columns
df.drop(columns=["EmployeeCount","Over18","EmployeeNumber","StandardHours","DailyRate","HourlyRate","JobLevel","MonthlyRate","StockOptionLevel","RelationshipSatisfaction",""],inplace=True)

# Feature Engineering Process
# Convert Categorical Values into Numerical values using Label Encoder

from sklearn.preprocessing import LabelEncoder
# Create a LabelEncoder instance


# Select categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Apply label encoding
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le             # store encoder for later


print("Categorical columns encoded successfully!")
print(df.head())

# Separate features and target
from sklearn.preprocessing import StandardScaler

x = df.drop(columns=['Attrition'])   # features
y = df['Attrition']                  # target

# Select only numeric columns from X
numeric_cols = x.select_dtypes(include=['int64', 'float64']).columns

# Initialize scaler
scaler = StandardScaler()

# Apply scaling only to numeric columns
x[numeric_cols] = scaler.fit_transform(x[numeric_cols])

# Combine back into final DataFrame (optional)
df_scaled = pd.concat([x, y], axis=1)

print(df_scaled.head())

# Split into training & testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Check Data Balance using PIE PLOT

activity_count = y.value_counts()
activity_count
# Show pie plot (Approach 1)
y.value_counts().plot.pie(autopct='%.2f')

# Apply SMOTE on training data only (NEVER on test data)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)

# You can check the class distribution after applying SMOTE
ax = y_train_sm.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Over-sampling using SMOTE")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 5. Train RandomForest

rf = RandomForestClassifier(random_state=20)
rf.fit(X_train_sm, y_train_sm)

# 6. Predictions
y_pred = rf.predict(x_test)

# 7. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import pickle

# Save the model
with open("rf_attrition_model.pkl", "wb") as f:
    pickle.dump(rf, f)

# Save the scaler (for numeric columns)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save the label encoder for each categorical column
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)