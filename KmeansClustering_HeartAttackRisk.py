import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Title
st.title("KMeans Clustering - Heart Attack Risk")

# Load dataset dari file lokal
try:
    df = pd.read_csv("heart_attack_prediction_dataset.csv")
except FileNotFoundError:
    st.error("❌ File 'heart_attack_prediction_dataset.csv' tidak ditemukan.")
    st.stop()
except Exception as e:
    st.error(f"❌ Terjadi error saat membaca file: {e}")
    st.stop()

# Tampilkan beberapa data awal
st.subheader("Data Awal")
st.write(df.head())

# Filter baris dengan nilai Heart Attack Risk = 0 jika kolom ada
if 'Heart Attack Risk' in df.columns:
    df = df[df['Heart Attack Risk'] != 0]

# Fitur tetap: Cholesterol dan Heart Rate
features = ['Cholesterol', 'Heart Rate']

# Cek apakah kolom tersedia
if not all(col in df.columns for col in features):
    st.error("❌ Kolom 'Cholesterol' atau 'Heart Rate' tidak ditemukan di dataset.")
    st.stop()

# Ambil data fitur
X = df[features]

# Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Slider untuk jumlah klaster
n_clusters = st.slider("Pilih jumlah klaster (K)", min_value=2, max_value=10, value=3)

# Clustering dengan KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Visualisasi
st.subheader("Visualisasi Klaster berdasarkan Cholesterol dan Heart Rate")
fig, ax = plt.subplots()
sns.scatterplot(
    x=X_scaled[:, 0],
    y=X_scaled[:, 1],
    hue=clusters,
    palette="Set1",
    ax=ax
)
ax.set_xlabel('Cholesterol (scaled)')
ax.set_ylabel('Heart Rate (scaled)')
st.pyplot(fig)

# Tampilkan hasil akhir
st.subheader("Data dengan Label Klaster")
st.write(df)
