import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.feature_selection import VarianceThreshold

st.set_page_config(page_title="Feature Explorer", layout="wide")
st.title("🔍 Feature Explorer Web App")

# File Upload
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data")
    st.dataframe(df.head())

    # Dataset Info
    st.subheader("📊 Dataset Info")
    st.write("Shape of dataset:", df.shape)
    st.write("Data types:")
    st.write(df.dtypes)

    # Missing Values Heatmap
    st.subheader("🧩 Missing Values Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    st.pyplot(fig)

    # Summary Stats
    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

    # Column Selection
    st.sidebar.header("Select Features")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_columns = st.sidebar.multiselect("Choose numeric features", numeric_cols, default=numeric_cols[:2])

    if selected_columns:
        st.subheader("📉 Feature Distribution")
        for col in selected_columns:
            fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

        # Boxplots
        st.subheader("📦 Box Plots (Outlier Detection)")
        for col in selected_columns:
            fig = px.box(df, y=col, title=f"Boxplot of {col}")
            st.plotly_chart(fig, use_container_width=True)

        # Correlation Heatmap
        st.subheader("🔗 Correlation Heatmap")
        corr = df[selected_columns].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Pairplot
        if len(selected_columns) <= 5:
            st.subheader("🔍 Pair Plot")
            fig = sns.pairplot(df[selected_columns])
            st.pyplot(fig)
        else:
            st.info("Select 5 or fewer features for Pair Plot to avoid overload.")

        # Feature Selection - Low Variance Filter
        st.subheader("🧠 Feature Selection (Low Variance Filter)")
        threshold = st.slider("Variance Threshold", 0.0, 1.0, 0.1)
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df[selected_columns].dropna())
        selected_features = [col for col, keep in zip(selected_columns, selector.get_support()) if keep]

        st.write("Features selected:", selected_features)
else:
    st.warning("Please upload a CSV file to get started.")
