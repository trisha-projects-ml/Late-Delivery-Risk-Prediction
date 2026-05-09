import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("APL_sample.csv")

# Drop columns
drop_columns = [
    "Customer Fname", "Customer Lname", "Customer Street",
    "Product Name", "Customer City", "Order City",
    "Customer Country", "Order Country",
    "Delivery Status", "Order Status"
]

df = df.drop(columns=[col for col in drop_columns if col in df.columns])

# Feature Engineering
df["Delay_Gap"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
df["Order_Complexity"] = df["Order Item Quantity"] * df["Order Item Product Price"]

# Encoding
df = pd.get_dummies(df, drop_first=True)

# Split
X = df.drop("Late_delivery_risk", axis=1)
y = df["Late_delivery_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# UI
st.title("Late Delivery Risk Prediction")

st.write("Enter order details:")

shipping_real = st.number_input("Days for shipping (real)", 1, 10)
shipping_scheduled = st.number_input("Days for shipment (scheduled)", 1, 10)
quantity = st.number_input("Order Item Quantity", 1, 10)
price = st.number_input("Product Price", 10.0, 500.0)

# Feature creation
delay_gap = shipping_real - shipping_scheduled
order_complexity = quantity * price

# Dummy input (simplified)
input_data = np.zeros(X.shape[1])

# Put values (basic approximation)
input_data[0] = shipping_real
input_data[1] = shipping_scheduled
input_data[2] = quantity
input_data[3] = price

# Predict
if st.button("Predict"):
    input_scaled = scaler.transform([input_data])
    prob = rf.predict_proba(input_scaled)[0][1]

    if prob < 0.3:
        risk = "Low Risk"
    elif prob < 0.7:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    st.write(f"Risk Probability: {prob:.2f}")
    st.write(f"Risk Category: {risk}")
