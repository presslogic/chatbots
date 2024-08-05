import streamlit as st

st.set_page_config(
    page_title="PL Chatbot",
    page_icon='💬',
    layout='wide'
)

st.header("PL Chatbot Implementations Demo")
st.write("支援短暫記憶buffer memory")
st.write("General Chatbot: 同時支援RAG和Web search工具的基本聊天機器人，由LLM判斷用什麼工具回答問題。")
st.write("RAG Chatbot: 使用RAG (vector search)回答問題的聊天機器人。")
st.write("Web search Chatbot: 使用Web search回答問題的聊天機器人。")