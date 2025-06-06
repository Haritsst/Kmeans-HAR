import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

    # Visualisasi
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
