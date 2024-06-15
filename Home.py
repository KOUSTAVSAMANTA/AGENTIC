import streamlit as st
from st_pages import Page, show_pages, add_page_title, Section
st.set_page_config(initial_sidebar_state="collapsed")

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
add_page_title("Home","🏠")
show_pages(
    [
        Section(name="Sections", icon="🏠"),
        Page("pages/chat.py", " - 💬 Chat"),
        Page("pages/Login & Register.py", " - 🔗 Login & Register"),
    ]
)
st.switch_page("pages/Login & Register.py")

