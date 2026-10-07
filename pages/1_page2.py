import streamlit as st
import pandas as pd

df = pd.read_csv("~/project/chrun_sd/data/raw_data.csv")
st.set_page_config(
    page_title="Dataset",
    page_icon="👋",
)

st.title("Dataset")
st.dataframe(df.head(5))
st.sidebar.success("Select a page above.")

if "my_input" not in st.session_state:
    st.session_state["my_input"] = ""

my_input = st.text_input("Input a text here", st.session_state["my_input"])
submit = st.button("Submit")
if submit:
    st.session_state["my_input"] = my_input
    st.write("You have entered: ", my_input)