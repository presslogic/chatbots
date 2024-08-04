from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
def config_llm():
    available_llms = ["gpt-4o-mini","gemini-pro"]
    llm_opt = st.sidebar.radio(
        label="LLM",
        options=available_llms,
        key="SELECTED_LLM"
        )
    if llm_opt == "gpt-4o-mini":
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, api_key=st.secrets["GOOGLE_API_KEY"])
    return llm

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v

def display_msg(msg, user):
    st.session_state.messages.append({"role": user, "content": msg})
    st.chat_message(user).write(msg)