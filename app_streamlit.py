import os
import warnings
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.documents import Document

warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Configuração da Página Web
st.set_page_config(page_title="SegurIA - Atendimento", page_icon="🛡️")
st.title("🛡️ SegurIA - Assistente Virtual")
st.caption("Converse com a documentação. As respostas são baseadas estritamente nos PDFs.")

# 2. Inicialização Segura do RAG (Usando session_state para evitar travamentos)
if "rag_chain" not in st.session_state:
    with st.spinner("Inicializando o modelo e lendo PDFs... (Isso pode levar alguns segundos na 1ª vez)"):
        os.makedirs("./docs", exist_ok=True)
        loader = PyPDFDirectoryLoader("./docs/")
        documentos = loader.load()

        if not documentos:
            st.warning("⚠️ Nenhum PDF encontrado na pasta 'docs'. Adicionando aviso provisório.")
            documentos = [Document(page_content="A base de conhecimento está vazia. Por favor, adicione PDFs na pasta 'docs'.")]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documentos)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        llm = ChatOllama(model="llama3.1", temperature=0.2)

        template = """
        Você é o assistente virtual de uma Seguradora.
        Responda à pergunta do usuário de forma educada e clara, utilizando APENAS o contexto fornecido abaixo.
        Se a informação não estiver no contexto, diga gentilmente que você não tem essa informação.

        Contexto da Seguradora (Extraído dos Manuais):
        {context}

        Pergunta do Segurado: {input}

        Resposta do Assistente:"""

        prompt = PromptTemplate.from_template(template)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        # Armazena o cérebro na sessão do usuário
        st.session_state.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 3. Memória do Chat (Session State)
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Exibe o histórico de mensagens
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Campo de Digitação e Processamento
if prompt_usuario := st.chat_input("Como posso ajudar? (ex: Meu carro quebrou)"):
    
    # Exibe a pergunta do usuário
    st.session_state.mensagens.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)

    # Exibe a resposta da IA
    with st.chat_message("assistant"):
        with st.spinner("Analisando contratos e apólices..."):
            try:
                resposta = st.session_state.rag_chain.invoke({"input": prompt_usuario})
                texto_resposta = resposta['answer']
            except Exception as e:
                texto_resposta = f"Ocorreu um erro ao processar a resposta: {str(e)}"
            
            st.markdown(texto_resposta)
            
    # Salva no histórico
    st.session_state.mensagens.append({"role": "assistant", "content": texto_resposta})