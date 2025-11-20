# Multi Model RAG - PDF AI Chat Application

A sophisticated Retrieval-Augmented Generation (RAG) application that enables intelligent conversations with multiple PDF documents using various AI models. Built with Streamlit, LangChain, and integrated with multiple language models for flexible and powerful document analysis.

## 🌟 Features

- **Multi-PDF Support**: Upload and process multiple PDF documents simultaneously
- **Multiple AI Models**: Choose from various AI models for different use cases:
  - **Gemini-1.5-Pro (API)**: Google's advanced generative AI model via API
  - **Gemma-2B**: Lightweight Google model running locally
  - **Mistral**: Open-source model for efficient responses
  - **Flan-T5-base**: Fine-tuned model for specialized tasks
- **Vector Storage Options**: Dual vector storage support with FAISS and ChromaDB
- **Interactive Chat Interface**: User-friendly chat interface with message history
- **Context-Aware Responses**: AI generates answers based on the content of uploaded PDFs
- **Knowledge Base Management**: Clear and reset vector database when needed

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemini models)
- Ollama installed (for local models like Mistral and Gemma)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/sanjithwoxsen/Multi_Model_RAG.git
   cd Multi_Model_RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Install Ollama (for local models)**
   
   Visit [Ollama's website](https://ollama.ai/) to install Ollama, then pull the required models:
   ```bash
   ollama pull mistral
   ollama pull gemma
   ```

## 💻 Usage

### Main Application (Recommended)

Run the main application with the enhanced chat interface:

```bash
streamlit run Main.py
```

This version includes:
- Chat history preservation
- Multiple model selection
- Knowledge base clearing functionality
- Real-time response generation

### Alternative Search UI

Run the simpler search-based interface:

```bash
streamlit run Search_UI.py
```

### How to Use

1. **Upload PDFs**: Use the sidebar to upload one or more PDF files
2. **Process Documents**: Click "Submit and Process" to extract text and create vector embeddings
3. **Select Model**: Choose your preferred AI model from the dropdown menu
4. **Ask Questions**: Type your questions in the chat input
5. **Get Answers**: Receive context-aware responses based on your PDF content

## 📋 Project Structure

```
Multi_Model_RAG/
├── Main.py              # Main Streamlit application with chat interface
├── Search_UI.py         # Alternative simple search interface
├── Rag.py              # Core RAG implementation and model classes
├── requirements.txt     # Python dependencies
├── .env                # Environment variables (API keys)
├── faiss_index/        # FAISS vector storage (created on first run)
└── chroma_db/          # ChromaDB vector storage (created on first run)
```

## 🤖 Available Models

### Gemini-1.5-Pro (API)
- **Type**: Cloud-based
- **Speed**: Fast
- **Quality**: Highest quality responses
- **Requirements**: Google API key

### Gemma-2B
- **Type**: Local
- **Speed**: Fast
- **Quality**: Good for general queries
- **Requirements**: Ollama with Gemma model

### Mistral
- **Type**: Local
- **Speed**: Moderate
- **Quality**: Excellent for detailed responses
- **Requirements**: Ollama with Mistral model

### Flan-T5-base (Fine-Tuned)
- **Type**: Local
- **Speed**: Very fast
- **Quality**: Good for specific tasks
- **Requirements**: Pre-trained model from Hugging Face

## 🔧 Technical Details

### Vector Storage

The application uses two vector storage systems:
- **FAISS**: Fast similarity search for efficient retrieval
- **ChromaDB**: Persistent vector database for long-term storage

### Text Processing

- **Chunk Size**: 10,000 characters
- **Chunk Overlap**: 3,000 characters
- **Embedding Model**: Google's embedding-001

### RAG Pipeline

1. PDF text extraction using PyPDF2
2. Text chunking with RecursiveCharacterTextSplitter
3. Vector embedding generation with Google's embedding model
4. Similarity search for relevant context
5. Context-aware response generation with selected AI model

## 🛠️ Dependencies

Key dependencies include:
- `streamlit`: Web interface
- `langchain` & `langchain-community`: RAG framework
- `transformers`: Hugging Face models
- `google-generativeai`: Google AI models
- `chromadb`: Vector database
- `faiss-cpu`: Similarity search
- `PyPDF2`: PDF processing

See `requirements.txt` for the complete list.

## ⚠️ Important Notes

- The `.env` file should never be committed to version control (add to `.gitignore`)
- First-time processing of PDFs may take longer as embeddings are created
- Local models require significant computational resources
- Clear the knowledge base when switching to a completely different set of documents

## 👨‍💻 Development

This project was developed by students of Woxsen University as part of their AI and Machine Learning coursework.

## 📝 License

This project is available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 🔒 Security

- Never commit API keys or sensitive credentials
- Keep your `.env` file private
- Use environment variables for all sensitive configuration

## 📞 Support

For issues and questions, please open an issue on the GitHub repository.

---

**Note**: This application requires active internet connection for Google API-based models and sufficient local resources for running local models.
