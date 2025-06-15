import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Title
st.title("KMeans Clustering - Heart Attack Risk")

# Load CSV langsung dari direktori proyek
try:
    df = pd.read_csv("heart_attack_prediction_dataset.csv")
except FileNotFoundError:
    st.error("❌ File 'heart_attack_prediction_dataset.csv' tidak ditemukan di direktori.")
    st.stop()
except Exception as e:
    st.error(f"❌ Terjadi error saat membaca file: {e}")
    st.stop()


if 'Blood Pressure' in df.columns:
    df['Blood Pressure'] = df['Blood Pressure'].astype(str).str.extract(r'(\d+)', expand=False).astype(int)


# Tampilkan data
st.subheader("Data Preview")
st.write(df.head())

# Hapus baris dengan Heart Attack Risk == 0 jika kolom tersedia
if 'Heart Attack Risk' in df.columns:
    df = df[df['Heart Attack Risk'] != 0]

df = df.select_dtypes(exclude=['object'])
binary_cols = [col for col in df.columns if df[col].nunique() == 2]
df = df.drop(columns=binary_cols)

# Pilih fitur
features = st.multiselect(
    "Pilih fitur untuk clustering",
    df.columns.tolist(),
    default=[] 
)

if features:
    X = df[features]

    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Slider jumlah klaster
    n_clusters = st.slider("Pilih jumlah klaster (K)", min_value=2, max_value=10, value=3)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Tambahkan label klaster ke dataframe
    df["Cluster"] = clusters

    # --- Dynamic Visualization Section ---
    # Case 1: Exactly two features are selected -> Use a standard scatter plot
    if len(features) == 2:
        st.subheader(f"Visualisasi Klaster: {features[0]} vs {features[1]}")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(
            x=X_scaled[:, 0], 
            y=X_scaled[:, 1], 
            hue=clusters, 
            palette="viridis", 
            ax=ax,
            s=100,
            alpha=0.7
        )
        ax.set_title(f"Cluster Visualization: {features[0]} vs {features[1]}", fontsize=16)
        ax.set_xlabel(f"Fitur: {features[0]} (Scaled)", fontsize=12)
        ax.set_ylabel(f"Fitur: {features[1]} (Scaled)", fontsize=12)
        ax.grid(True)
        st.pyplot(fig)

    # Case 2: More than two features are selected -> Use PCA
    elif len(features) > 2:
        st.subheader("Visualisasi Klaster dengan PCA")
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame(
            data=principal_components, 
            columns=['Principal Component 1', 'Principal Component 2']
        )
        pca_df['Cluster'] = clusters

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(
            x='Principal Component 1', y='Principal Component 2', hue='Cluster', 
            palette="viridis", data=pca_df, ax=ax, s=100, alpha=0.7
        )
        
        ax.set_title("Cluster Visualization with PCA", fontsize=16)
        ax.set_xlabel("Principal Component 1", fontsize=12)
        ax.set_ylabel("Principal Component 2", fontsize=12)
        ax.grid(True)
        st.pyplot(fig)
        
        explained_variance = pca.explained_variance_ratio_
        st.info(f"""
        **Explained Variance:**
        - **Principal Component 1:** {explained_variance[0]:.2%}
        - **Principal Component 2:** {explained_variance[1]:.2%}
        - **Total Variance Explained by 2 Components:** {sum(explained_variance):.2%}
        """)

        st.subheader("Deskripsi Kontribusi Fitur pada Komponen Utama (PCA)")
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=['Principal Component 1', 'Principal Component 2'],
            index=features
        )
        st.markdown("Bobot berikut menunjukkan seberapa kuat setiap fitur memengaruhi setiap komponen utama. **Semakin Besar Nilainya** berarti pengaruhnya lebih besar.")
        st.dataframe(loadings_df.style.background_gradient(cmap='viridis'))
        
    # Case 3: Less than two features selected
    else:
        st.warning("Pilih minimal 2 fitur untuk membuat visualisasi plot 2D.")
        
    # Deskripsi klaster
    st.subheader("Deskripsi Statistik per Klaster (Per Fitur)")

    for feature in features:
        st.markdown(f"#### Statistik untuk Fitur: `{feature}`")
        desc = df.groupby("Cluster")[feature].describe()
        st.write(desc)

    # Tampilkan hasil clustering
    st.subheader("Data dengan Label Klaster")
    st.write(df)
else:
    st.warning("Pilih minimal satu fitur untuk proses clustering.")
