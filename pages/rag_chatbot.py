from langchain_openai import OpenAIEmbeddings
import utils
import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('RAG Chatbot')

class RAGChatbot:

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
    
    @st.spinner('Analyzing documents..')
    def setup_retrieval_chain(self, files):
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
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        # embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
        splits = text_splitter.split_documents(docs)
        db = FAISS.from_documents(splits, embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        template = """Answer the question: {question} based only on the following context:
            context: {context}
        """
        prompt = PromptTemplate.from_template(template = template,
                                input_varaibles = ["context", "question"])
        output_parser = StrOutputParser()
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        retrieval_chain = (
            {"context": retriever | format_docs,  "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | output_parser
        )
        return retrieval_chain
    
    def main(self):
        ctx = get_script_run_ctx()
        session_id = ctx.session_id
        st.write(f"Session ID: {session_id}")
        uploaded_files = st.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
        if uploaded_files:
            if "retrieval_chain" not in st.session_state:
                retrieval_chain = self.setup_retrieval_chain(uploaded_files)
                st.session_state['retrieval_chain'] = retrieval_chain
                retrieval_chain = st.session_state["retrieval_chain"]
            else:
                 retrieval_chain = st.session_state["retrieval_chain"]
        else:
            st.error("Please upload PDF documents to continue!")
            st.stop()
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
                result = retrieval_chain.invoke(
                    user_query
                )
            response = result
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

if __name__ == "__main__":
    obj = RAGChatbot()
    obj.main()