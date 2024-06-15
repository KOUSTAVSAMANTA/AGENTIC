import streamlit as st
from passlib.hash import pbkdf2_sha256
from pymongo import MongoClient

from st_pages import Page, show_pages, add_page_title, Section
import clipboard

import os

st.set_page_config(initial_sidebar_state="collapsed")
show_pages(
    [
        Section(name="Sections", icon="🏠"),
        Page("pages/chat.py", " - 💬 Chat"),
        Page("pages/Login & Register.py", " - 🔗 Login & Register"),
    ]
)
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
# Custom CSS styles
custom_css = """
<style>
/* Add your custom CSS styles here */
body {
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
}

.container {
    max-width: 600px;
    margin: 0 auto;
    padding: 20px;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

.header {
    text-align: center;
    margin-bottom: 20px;
}

.input-container {
    margin-bottom: 15px;
}

.input-container input {
    width: 100%;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box;
}

.button-container {
    text-align: center;
}

.button-container button {
    padding: 10px 20px;
    background-color: #007bff;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.button-container button:hover {
    background-color: #0056b3;
}

.error-message {
    color: #ff0000;
    margin-top: 10px;
    text-align: center;
}
</style>
"""

# Render custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

userID = ""


def on_copy_click(text):
    st.session_state.copied.append(text)
    clipboard.copy(text)


if "copied" not in st.session_state:
    st.session_state.copied = []

client = MongoClient("mongodb+srv://koustav:koustav2003@cluster0.wmnc2.mongodb.net/?retryWrites=true&w=majority",
                     ssl=True
                     )
db = client['AUTH']
collection = db['Authentication']


# Streamlit UI
def is_username_duplicate(username):
    return collection.find_one({"username": username}) is not None


def sign_up():
    # st.subheader("Sign Up")
    username = st.text_input("Username", key="signup_username")
    password = st.text_input("Password", type="password", key="signup_password")

    if st.button("Sign Up"):
        if is_username_duplicate(username):
            st.error("Username already exists. Please choose a different username.")
        else:
            hashed_password = pbkdf2_sha256.hash(password)
            user_data = {"username": username, "password": hashed_password}
            collection.insert_one(user_data)
            st.success("User created successfully! Please sign in.")


def sign_in():
    # st.subheader("Sign In")
    username = st.text_input("Username", key="signin_username")
    password = st.text_input("Password", type="password", key="signin_password")

    if st.button("Sign In"):
        user_data = collection.find_one({"username": username})
        if user_data and pbkdf2_sha256.verify(password, user_data['password']):
            st.success(f"Welcome, {username}!")
            print("entered")

            st.session_state.username = username
            global userID
            userID = username
            print("set")
            # print(st.session_state.choice2)
            st.session_state.choice2 = "chat"
            st.switch_page("./pages/chat.py")
        else:
            st.error("Invalid credentials. Please try again.")


# Main content
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Sign Up or Sign In")
tab1, tab2 = st.tabs(["Sign Up", "Sign In"])

with tab1:
    st.header("Sign Up")
    sign_up()

with tab2:
    st.header("Sign In")
    sign_in()
