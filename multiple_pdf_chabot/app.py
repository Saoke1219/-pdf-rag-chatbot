import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama as OllamaLLM
from langchain_ollama import OllamaEmbeddings
from htmlTemplates import css, bot_template, user_template
import re
import os
import time


VECTORSTORE_PATH = "vectorstore"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    st.write(f"✅ Document split into {len(chunks)} chunks.")
    return chunks


def get_vectorstore(text_chunks, embeddings):
    start = time.time()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    st.info(f"✅ Embedding and indexing completed in {time.time() - start:.2f} seconds.")
    return vectorstore


def load_cached_vectorstore(embeddings):
    if os.path.exists(VECTORSTORE_PATH):
        try:
            return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning("⚠️ Could not load cached vectorstore. Reprocessing required.")
    return None


def get_conversation_chain(vectorstore):
    try:
        llm = OllamaLLM(model="phi3:3.8b", temperature=0.5)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
    except Exception as e:
        st.error(f"❌ Error creating conversation chain: {str(e)}")
        return None


def handle_userinput(user_question):
    try:
        start_time = time.time()
        response = st.session_state.conversation({'question': user_question})
        end_time = time.time()
        latency = end_time - start_time
        st.success(f"✅ Answered in {latency:.2f} seconds")

        st.session_state.chat_history = response['chat_history']

        last_pair = st.session_state.chat_history[-2:]
        for msg in last_pair:
            tpl = user_template if msg == last_pair[0] else bot_template
            st.write(tpl.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        if 'source_documents' in response:
            with st.expander("📚 Source References"):
                for doc in response['source_documents']:
                    st.markdown("**Reference:**")
                    st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

    except Exception as e:
        st.error(f"❌ Error processing your question: {str(e)}")


def main():
    load_dotenv()
    st.set_page_config(page_title="Zizi Chatbot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Zizi Chatbot :books:")

    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_input("Ask a question about Zizi:")
        submitted = st.form_submit_button("Send")

    if submitted and user_question:
        handle_userinput(user_question)

    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("🔄 Processing..."):
                try:
                    embeddings = OllamaEmbeddings(model="nomic-embed-text")
                    raw_text = get_pdf_text(pdf_docs)
                    cleaned_text = clean_text(raw_text)
                    text_chunks = get_text_chunks(cleaned_text)

                    # ✅ Show preview
                    with st.expander("🔍 Preview Text Chunks (first 20 shown)"):
                        for i, chunk in enumerate(text_chunks[:20]):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.write(chunk)

                    vectorstore = get_vectorstore(text_chunks, embeddings)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("✅ Documents processed and conversation ready!")

                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")

        if st.button("Load Cached Index"):
            with st.spinner("🔄 Loading saved vectorstore..."):
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                vectorstore = load_cached_vectorstore(embeddings)
                if vectorstore:
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("✅ Cached index loaded!")

if __name__ == '__main__':
    main()