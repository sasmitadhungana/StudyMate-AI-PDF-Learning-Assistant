📘 StudyMate AI – Intelligent PDF Learning Assistant

StudyMate AI is an AI-powered educational application that transforms static PDF documents into interactive learning companions. Users can upload educational PDFs (textbooks, notes, research papers) and ask questions 
in natural language to receive context-aware answers with page-level citations, enabling efficient and smart learning.

🚀 Key Features

🤖 AI-Powered Question Answering from PDF documents

📄 Context-Aware Responses with page number citations

🧠 Powered by Groq AI (Llama 3.3 70B)

🖥️ Simple & Interactive Streamlit Web Interface

📖 Learning-oriented explanations for students

🛠️ Technology Stack
Layer	Technology
Frontend	Streamlit
Backend	Python 3.9+
AI Engine	Groq Cloud API
Language Model	Llama 3.3 70B
PDF Processing	PyPDF2
API Communication	Requests

⚙️ Installation & Setup
# Clone the repository
git clone https://github.com/sasmitadhungana/StudyMate-AI-PDF-Learning-Assistant.git
cd StudyMate-AI-PDF-Learning-Assistant

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/frontend/app.py

🔑 API Key Configuration

Visit 👉 https://console.groq.com

Create a free account

Generate a Groq API key (gsk_...)

Enter the API key in the application sidebar

📖 How to Use

Upload PDF documents
Allow the system to extract and analyze text
Ask questions in natural language
Receive AI-generated answers with page citations

Example Questions:
“Summarize chapter 2”
“Explain supervised learning”
“What are the key findings?”
“Give examples from the document”

📂 Project Structure
StudyMate-AI-PDF-Learning-Assistant/
├── src/
│   ├── frontend/
│   │   └── app.py
│   ├── core/
│   │   ├── pdf_processor.py
│   │   ├── qa_engine.py
│   │   └── groq_service.py
├── requirements.txt
├── README.md
└── LICENSE

🎯 Use Case
Students studying from digital notes or textbooks
Researchers analyzing academic papers
Self-learners seeking interactive explanations from PDFs

⭐ Acknowledgments
Groq Inc. for high-performance AI inference
Streamlit for the interactive UI framework
Open-source community for essential tools

If you find this project helpful, please star ⭐ the repository
Contributions, issues, and suggestions are welcome.
