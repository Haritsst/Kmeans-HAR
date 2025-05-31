import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Title
st.title("KMeans Clustering - Heart Attack Risk")

# Upload file
uploaded_file = st.file_uploader("heart_attack_prediction_dataset.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.write(df.head())

    # Select relevant columns (you can customize this)
    features = st.multiselect("Pilih fitur untuk clustering", df.columns.tolist(), default=df.columns.tolist())

    if features:
        X = df[features]

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Slider for number of clusters
        n_clusters = st.slider("Pilih jumlah klaster (K)", min_value=2, max_value=10, value=3)

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # Add cluster labels to dataframe
        df["Cluster"] = clusters

        # Plot
        st.subheader("Visualisasi Clustering")
        if len(features) >= 2:
            fig, ax = plt.subplots()
            sns.scatterplot(
                x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette="Set1", ax=ax
            )
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            st.pyplot(fig)
        else:
            st.warning("Pilih minimal 2 fitur untuk visualisasi 2D.")

        # Show dataframe with clusters
        st.subheader("Data dengan Label Klaster")
        st.write(df)
