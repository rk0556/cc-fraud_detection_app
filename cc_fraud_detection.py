import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Function to detect fraudulent transactions
def detect_fraudulent_transactions(new_data, trained_model, scaler):
    # Standardize the new data using the provided scaler
    new_data_scaled = scaler.transform(new_data)

    # Predict anomalies using the trained model
    predictions = trained_model.predict(new_data_scaled)

    # Convert predictions to binary format (1 for anomalies, 0 for normal)
    predictions = [1 if x == -1 else 0 for x in predictions]

    # Return the predictions
    return predictions

# Streamlit app
def main():
    st.title("Credit Card Fraud Detection App")
    
    st.write("""
    Upload a new set of credit card transactions to detect potential fraudulent transactions.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write("Uploaded Data:")
        st.write(data.head())

        # Check if necessary columns are in the uploaded file
        if {'Time', 'Amount'}.issubset(data.columns):
            # Select features and apply scaling
            features = data.drop(['Time'], axis=1)  # Drop 'Time' column
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)

            # Train Isolation Forest model
            iso_forest = IsolationForest(contamination=0.02, random_state=42)
            iso_forest.fit(X_scaled)

            # Detect anomalies
            predictions = detect_fraudulent_transactions(features, iso_forest, scaler)
            
            # Add predictions to the data
            data['Anomaly'] = predictions

            # Display the results
            st.write("Results:")
            st.write(data.head())

            # Visualize the anomalies
            st.write("Anomaly Visualization:")
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(X_scaled)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['Anomaly'], cmap='coolwarm', alpha=0.6)
            plt.title('Anomalies in PCA Space')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            st.pyplot(plt)
        else:
            st.error("The uploaded file does not contain the required columns.")
        
if __name__ == "__main__":
    main()
