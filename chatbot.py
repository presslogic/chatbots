import utils
import streamlit as st
from streaming import StreamHandler
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
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('PL Chatbot')

class BasicChatbot:

    def __init__(self):
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
    
    @st.spinner('Analyzing documents..')
    def setup_retriever_tool(self, files, file_description):
        docs = []
        for file in files:
            file_path = self.save_file(file)
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )
        model_name = "BAAI/bge-m3"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        # embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
        splits = text_splitter.split_documents(docs)
        db = FAISS.from_documents(splits, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 5})
        qa_llm_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever)
        langchain_tool = Tool(
            name=file_description,
            func=qa_llm_chain.run,
            description=f'You must use this tool to answer questions about {file_description}'
        )
        return langchain_tool
    
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
        st.sidebar.write("Upload PDF file and write a description for the file")
        uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
        file_description = st.sidebar.text_input("Enter a description for the uploaded file")
        if uploaded_files and file_description:
            if "retrieval_tool" not in st.session_state:
                retrieval_tool = self.setup_retriever_tool(uploaded_files, file_description)
                st.session_state['retrieval_tool'] = retrieval_tool
                st.session_state["agent"] = self.setup_chain(new_tool=retrieval_tool)
                agent = st.session_state["agent"]
            else:
                 agent = st.session_state["agent"]
        else:
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
                result = agent.invoke(
                    {"input":user_query},
                    config={"configurable": {"session_id": session_id}}
                )
            response = result["output"]
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

if __name__ == "__main__":
    obj = BasicChatbot()
    obj.main()