import utils
import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
import os
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from datetime import datetime
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Deepseek Chatbot')

class BasicChatbot:

    def __init__(self):
        utils.clear_session_button()
        utils.sync_st_session()
        self.llm = utils.config_llm()
        if 'llm' not in st.session_state:
            st.session_state.clear()
            st.session_state['llm'] = self.llm.__class__.__name__
        elif st.session_state['llm'] is not self.llm.__class__.__name__:
            st.session_state.clear()
            st.session_state['llm'] = self.llm.__class__.__name__
    
    def setup(_self):
        memory = ConversationBufferMemory(return_messages=True)
        prompt_template = PromptTemplate(
            input_variables=["history", "input"],
            template="{history}\nUser: {input}\nAssistant:"
        )
        chain = ConversationChain(
            llm=_self.llm,
            memory=memory,
            verbose=True,
            prompt=prompt_template
        )
        return chain
    
    def extract_thinking_and_response(_self, text: str) -> tuple[str, str]:
        think_start = text.find("<think>")
        think_end = text.find("</think>")
        
        if think_start != -1 and think_end != -1:
            thinking = text[think_start + 7:think_end].strip()
            response = text[think_end + 8:].strip()
            return thinking, response
        return "", text
    
    def main(self):
        ctx = get_script_run_ctx()
        session_id = ctx.session_id
        st.write(f"Session ID: {session_id}")
        if "agent" not in st.session_state:
            st.session_state["agent"] = self.setup()
            agent = st.session_state["agent"]
        else:
            agent = st.session_state["agent"]
        user_query = st.chat_input(placeholder="Ask me anything!")
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])
        if user_query:
            utils.display_msg(user_query, 'user')
            # with st.chat_message("assistant"):
            # st_cb = StreamHandler(st.empty())
            with st.spinner('Typing...'):
                result = agent.invoke(
                    {"input":user_query},
                    config={"configurable": {"session_id": session_id}}
                )
            response = result['response']
            thinking, response = self.extract_thinking_and_response(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            if thinking:
                with st.expander("Show thinking process", expanded=False):
                    st.markdown(thinking)
            st.chat_message("assistant").write(response)

if __name__ == "__main__":
    obj = BasicChatbot()
    obj.main()