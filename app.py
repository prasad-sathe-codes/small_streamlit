import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    return pd.read_csv(url, header=None, names=columns)

# Main function to create the app
def main():
    st.title("Iris Dataset Visualization")
    st.write("Explore the Iris dataset with scatter plots.")

    # Load data
    data = load_data()
    st.dataframe(data)

    # User input for plot features
    x_axis = st.selectbox("Select X-axis feature", data.columns[:-1])
    y_axis = st.selectbox("Select Y-axis feature", data.columns[:-1])

    # Plot
    if st.button("Generate Plot"):
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x_axis, y=y_axis, hue="class", ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
