import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="RASITH Data Prediction App", layout="wide")

# --- TITLE ---
st.title("📊 RASITH Streamlit Data Prediction App")
st.write("""
Welcome!  
This web app allows you to upload a dataset (CSV file), visualize it,
and build a simple Linear Regression model for predictions.
""")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.write(data.head())

    # --- BASIC INFO ---
    st.subheader("📈 Dataset Summary")
    st.write(data.describe())

    # --- COLUMN SELECTION ---
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_columns) >= 2:
        x_col = st.selectbox("Select X (Independent variable):", numeric_columns)
        y_col = st.selectbox("Select Y (Target variable):", numeric_columns)

        if x_col and y_col and x_col != y_col:
            X = data[[x_col]]
            y = data[y_col]

            # --- TRAIN MODEL ---
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # --- RESULTS ---
            st.subheader("🧠 Model Results")
            st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")
            st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.3f}")

            # --- PLOT RESULTS ---
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, color='blue')
            ax.plot(y_test, y_test, color='red', linewidth=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

            # --- PREDICTION INPUT ---
            st.subheader("🔮 Try a Prediction")
            input_val = st.number_input(f"Enter a value for {x_col}:", float(X[x_col].min()), float(X[x_c_]()_]()_
streamlit
pandas
numpy
scikit-learn
matplotlib
