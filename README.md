# seguria-rag-assistant
SegurIA: Local RAG-based Insurance Assistant
SegurIA is a specialized virtual assistant designed to provide accurate, context-aware information regarding insurance policies and manuals. By leveraging Retrieval-Augmented Generation (RAG), the system ensures that responses are grounded strictly in the provided documentation (PDFs), eliminating hallucinations and ensuring data privacy.

🚀 Key Features
Privacy-First Architecture: The entire pipeline runs locally on WSL2 (Ubuntu) using Ollama. No sensitive insurance data or PII (Personally Identifiable Information) ever leaves the local environment.

Contextual Intelligence: Utilizes the Llama 3.1 model to process complex insurance clauses and provide human-like explanations.

Dual Interface Support: Offers flexibility through two distinct web interfaces:

Streamlit: A sleek, modern UI integrated into the existing agent ecosystem.

Gradio: A robust, high-reliability alternative for rapid testing and diagnostics.

Dynamic Knowledge Base: Automatically parses and indexes PDF documents (policies, FAQs, and manuals) located in the /docs directory.

🛠️ Technical Stack
LLM Engine: Ollama running Llama 3.1.

Orchestration: LangChain for RAG pipeline management.

Vector Database: ChromaDB for high-performance similarity search.

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (HuggingFace).

Environment: WSL2 / Ubuntu / Python 3.10+.

📖 How it Works (RAG Pipeline)
Ingestion: The system scans the ./docs folder for PDF files using PyPDFDirectoryLoader.

Chunking: Documents are split into smaller segments using RecursiveCharacterTextSplitter to maintain semantic context.

Vectorization: Text chunks are converted into mathematical vectors (embeddings) and stored in ChromaDB.

Retrieval: When a user asks a question, the system searches the database for the most relevant document segments.

Generation: The LLM receives the user's query alongside the retrieved segments to generate a precise, evidence-based response.

⚙️ Setup & Installation
Clone the repository:

Bash
git clone https://github.com/your-username/seguria-rag.git
cd seguria-rag
Set up the environment:

Bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Run the application:

Bash
# For Streamlit
streamlit run app_streamlit.py

# For Gradio
python app_gradio.py
