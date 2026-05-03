import os
import warnings
import gradio as gr
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

# 1. Cria a pasta docs caso não exista
os.makedirs("./docs", exist_ok=True)

print("[1/4] Lendo arquivos PDF da pasta './docs'...")
loader = PyPDFDirectoryLoader("./docs/")
documentos = loader.load()

# Prevenção de erro: Se a pasta estiver vazia
if not documentos:
    print("⚠️ ATENÇÃO: Nenhum PDF encontrado. Adicionando documento de aviso provisório.")
    documentos = [Document(page_content="A base de conhecimento está vazia. Por favor, adicione PDFs na pasta 'docs'.")]

# 2. Processamento e Banco Vetorial
print("[2/4] Quebrando texto e processando Embeddings...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documentos)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Configuração do Llama 3.1
print("[3/4] Conectando ao Llama 3.1 local via Ollama...")
llm = ChatOllama(model="llama3.1", temperature=0.2)

template = """
Você é o assistente virtual de uma Seguradora.
Responda à pergunta do usuário de forma educada e clara, utilizando APENAS o contexto fornecido abaixo.
Se a informação não estiver no contexto, diga gentilmente que você não tem essa informação.

Contexto da Seguradora (Extraído dos PDFs):
{context}

Pergunta do Segurado: {input}

Resposta do Assistente:"""

prompt = PromptTemplate.from_template(template)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 4. Função de Resposta para a Interface
def responder(mensagem, historico):
    try:
        resposta = rag_chain.invoke({"input": mensagem})
        return resposta['answer']
    except Exception as e:
        return f"Erro ao processar a resposta: {str(e)}"

# 5. Configuração da Interface Web Gradio
print("[4/4] Iniciando Servidor Web Gradio...")
demo = gr.ChatInterface(
    fn=responder, 
    title="🛡️ SegurIA - Assistente RAG de Seguros",
    description="Converse com a documentação. As respostas são baseadas estritamente nos manuais e apólices em PDF localizados na pasta `docs`."
)

# Executa o servidor na porta que já testamos e funcionou
demo.launch(server_name="0.0.0.0", server_port=8505)