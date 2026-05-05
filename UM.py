import pandas as pd 
import numpy as np
df = pd.read_excel("/Users/saibesh/Desktop/Study Material/UM/APL_Logistics.xlsx")
print(df.head())
print(df.info())
print(df.isnull().sum())
df = df.drop([
    "Customer Fname", "Customer Lname", "Customer Street",
    "Product Name", "Customer City", "Order City",
    "Customer Country", "Order Country",
    "Delivery Status", "Order Status"
], axis=1)

df["Delay_Gap"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]

df["Order_Complexity"] = df["Order Item Quantity"] * df["Order Item Product Price"]

df = pd.get_dummies(df, drop_first=True)
X = df.drop("Late_delivery_risk", axis=1)
y = df["Late_delivery_risk"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# 👇 ADD THIS WHOLE BLOCK BELOW

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
from sklearn.metrics import roc_auc_score

y_prob = rf.predict_proba(X_test)[:,1]
def risk_category(p):
    if p < 0.3:
        return "Low Risk"
    elif p < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

risk_labels = [risk_category(p) for p in y_prob]

print(risk_labels[:10])  # just to see first 10 results

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

from sklearn.metrics import accuracy_score, classification_report

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
import pandas as pd

importance = pd.Series(rf.feature_importances_, index=X.columns)
print("Top 5 Important Features:")
print(importance.sort_values(ascending=False).head(5))

