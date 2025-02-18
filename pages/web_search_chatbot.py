import utils
import streamlit as st
from streamlit.scriptrunner.script_run_context import get_script_run_ctx
import os
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from datetime import datetime
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Web search chatbot')

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

    def save_file(self, file):
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path
    
    def setup_chain(_self, new_tool=None):
        datetime_tool = Tool(
            name="return the current datetime",
            func=lambda x: datetime.now().isoformat(),
            description="Returns the current datetime",
        )
        os.environ["SERPAPI_API_KEY"] = st.secrets["SERPAPI_API_KEY"]
        tools = load_tools(["serpapi"], llm=_self.llm)
        tools.append(datetime_tool)
        if new_tool:
            tools.append(new_tool)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent = initialize_agent(
            tools,
            _self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True,
            memory=memory,
            output_key = "result",
            max_iterations=10,
        )
        return agent
    
    def main(self):
        ctx = get_script_run_ctx()
        session_id = ctx.session_id
        st.write(f"Session ID: {session_id}")
        if "agent" not in st.session_state:
            st.session_state["agent"] = self.setup_chain()
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
                result = agent(
                   user_query
                )
            response = result["output"]
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

if __name__ == "__main__":
    obj = BasicChatbot()
    obj.main()