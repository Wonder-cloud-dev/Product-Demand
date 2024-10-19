import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load and clean the data
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")
data = data.dropna()  # Remove any missing values

# Prepare data for the model
X = data[["Total Price", "Base Price"]]  # Features (input data)
y = data["Units Sold"]  # Target (what we want to predict)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Decision Tree)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Streamlit app interface
st.title("Simple Product Demand Prediction")

# Inputs for making a prediction
total_price = st.number_input("Enter Total Price:", min_value=0.0)  # Total price input
base_price = st.number_input("Enter Base Price:", min_value=0.0)    # Base price input

# Button to predict units sold
if st.button("Predict Units Sold"):
    # Use the model to predict based on the input prices
    prediction = model.predict(np.array([[total_price, base_price]]))
    st.write(f"Predicted Units Sold: {prediction[0]:.2f}")

# Button to show scatter plot
if st.button("Show Scatter Plot"):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=data["Units Sold"], y=data["Total Price"])  # Simple scatter plot
    st.pyplot(plt)

# Button to show heatmap
if st.button("Show Correlation Heatmap"):
    plt.figure(figsize=(6,4))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")  # Correlation heatmap
    st.pyplot(plt)
