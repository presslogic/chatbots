from langchain_ollama import ChatOllama
import streamlit as st
url = "https://ollama.medialens.io/"
def config_llm():
    available_llms = ["deepseek-r1:14b","deepseek-r1:8b","deepseek-r1:7b"]
    llm_opt = st.sidebar.radio(
        label="LLM",
        options=available_llms,
        key="SELECTED_LLM"
        )
    if llm_opt == "deepseek-r1:14b":
        llm = ChatOllama(model="deepseek-r1:14b", base_url=url)
    elif llm_opt == "deepseek-r1:8b":
        llm = ChatOllama(model="deepseek-r1:8b", base_url=url)
    elif llm_opt == "deepseek-r1:7b":
        llm = ChatOllama(model="deepseek-r1:7b", base_url=url)
    return llm

def clear_session_button():
    st.sidebar.button("Clear session", on_click=lambda: st.session_state.clear())

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v

def display_msg(msg, user):
    st.session_state.messages.append({"role": user, "content": msg})
    st.chat_message(user).write(msg)