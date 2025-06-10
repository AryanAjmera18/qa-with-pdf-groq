import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Disable watchdog warning (fixes torch.classes error)
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

# Load environment variables
load_dotenv()

# Streamlit UI layout
st.set_page_config(page_title="PDF Chatbot with Memory", layout="wide")
st.title("📚 Chat with Your PDFs")
st.markdown("Upload PDFs, ask questions, and get contextual answers with memory 💬")

# Sidebar for configuration
with st.sidebar:
    st.header("🔐 API & Session Settings")
    api_key = st.text_input("Enter your **Groq API key**:", type="password")
    session_id = st.text_input("Session ID", value="default_session")

    st.markdown("---")
    st.markdown("🗂️ Upload your PDF files below")

# Only continue if API key is provided
if not api_key:
    st.warning("⚠️ Please enter your Groq API key to continue.")
    st.stop()

# Load LLM
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

# Initialize embedding model (with fix)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Upload and process PDFs
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    with st.spinner("⏳ Reading and processing PDFs..."):
        for uploaded_file in uploaded_files:
            temp_pdf_path = f"./temp_{uploaded_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()
            documents.extend(docs)

        # Text splitting & embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Contextual question prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question "
                       "which might reference context in the chat history, "
                       "formulate a standalone question. Do NOT answer the question."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # QA system prompt
        system_prompt = (
            "You are a helpful assistant for question-answering over PDF documents. "
            "Use the retrieved context to answer the question in 3 sentences or less. "
            "Say 'I don't know' if the answer isn't available.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Session-level chat history
        if 'store' not in st.session_state:
            st.session_state.store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User chat input
        st.markdown("### 💬 Ask a question about the uploaded PDFs")
        user_input = st.text_input("Your question:")

        if user_input:
            with st.spinner("🤖 Generating response..."):
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                st.success(response["answer"])

            # Optional: Expand to show chat history
            with st.expander("🕘 Chat History"):
                for msg in session_history.messages:
                    role = "🧑‍💻 You" if msg.type == "human" else "🤖 Assistant"
                    st.markdown(f"**{role}:** {msg.content}")

else:
    st.info("📄 Upload PDF file(s) to start chatting.")
