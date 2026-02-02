"""
AI-POWERED PDF QA SYSTEM - UPDATED MODELS
"""
import streamlit as st
import PyPDF2
import io
import requests
import json

st.set_page_config(
    page_title="AI PDF QA System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Powered PDF QA System")
st.markdown("**Intelligent answers using Groq AI**")
st.markdown("---")

# Available models (updated)
AVAILABLE_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B (Recommended)",
    "llama-3.1-8b-instant": "Llama 3.1 8B Instant",
    "llama-3.2-3b-preview": "Llama 3.2 3B Preview",
    "gemma2-9b-it": "Gemma2 9B",
    "mixtral-8x7b-32768": "Mixtral (Legacy)"
}

# Initialize session state
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "llama-3.3-70b-versatile"

# Sidebar for configuration
with st.sidebar:
    st.header("🔑 Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Enter your Groq API key",
        help="Get from console.groq.com",
        key="api_key_input"
    )
    
    if api_key:
        st.session_state.api_key = api_key
        if len(api_key) > 20:
            st.success("✅ API Key configured")
    
    # Model selection
    st.divider()
    st.header("🤖 AI Model")
    
    selected_model = st.selectbox(
        "Choose AI Model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: AVAILABLE_MODELS[x],
        index=0
    )
    
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.info(f"Selected: {AVAILABLE_MODELS[selected_model]}")
    
    st.divider()
    
    # PDF Upload
    st.header("📁 Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Choose PDF file",
        type=["pdf"],
        key="pdf_uploader"
    )
    
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")
        
        if st.button("🔬 Process PDF for AI Q&A", type="primary", use_container_width=True):
            with st.spinner("Extracting text from PDF..."):
                try:
                    # Extract all text from PDF
                    pdf_bytes = uploaded_file.getvalue()
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                    
                    all_text = ""
                    for i in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[i]
                        text = page.extract_text()
                        if text.strip():  # Only add if there's text
                            all_text += f"\n\n--- Page {i+1} ---\n{text}"
                    
                    st.session_state.pdf_text = all_text
                    st.session_state.processed = True
                    st.session_state.filename = uploaded_file.name
                    st.session_state.page_count = len(pdf_reader.pages)
                    
                    st.success(f"📚 PDF processed! {len(pdf_reader.pages)} pages")
                    st.info("✅ Ready for AI questions")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {e}")

# Function to call Groq API with current models
def ask_groq_ai(question, pdf_text, api_key, model):
    """Use Groq AI to answer questions based on PDF content"""
    
    # Limit context to avoid token limits
    if len(pdf_text) > 6000:
        pdf_context = pdf_text[:6000] + "... [truncated]"
    else:
        pdf_context = pdf_text
    
    # Prepare the prompt
    prompt = f"""You are an AI educational assistant. Answer questions based ONLY on the provided PDF content.

PDF CONTENT:
{pdf_context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based SOLELY on the PDF content above
2. If the answer cannot be found in the PDF, say so clearly
3. Be educational, clear, and concise
4. Reference specific pages when possible
5. Explain concepts in simple terms

ANSWER:"""
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful educational assistant for PDF documents."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            error_msg = f"API Error {response.status_code}"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg = f"{error_data['error'].get('message', error_msg)}"
            except:
                pass
            return f"**API Error:** {error_msg}\n\nTry a different model or check your API key."
            
    except requests.exceptions.Timeout:
        return "**Error:** Request timeout. The model might be busy. Try again or use a smaller model."
    except Exception as e:
        return f"**Error:** {str(e)}"

# Main content
if st.session_state.processed and st.session_state.pdf_text:
    # Show document info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Document", st.session_state.get('filename', 'PDF'))
    with col2:
        st.metric("Pages", st.session_state.get('page_count', 0))
    with col3:
        status = "AI Ready" if st.session_state.api_key else "Demo Mode"
        st.metric("Status", status)
    with col4:
        st.metric("Model", AVAILABLE_MODELS.get(st.session_state.selected_model, "AI"))
    
    st.markdown("---")
    
    # Question answering section
    st.header("💬 Ask AI About This PDF")
    
    if not st.session_state.api_key:
        st.warning("⚠️ **Demo Mode** - Add API key in sidebar for real AI answers")
        st.info("Get a free API key from [console.groq.com](https://console.groq.com)")
    
    # Question input
    question = st.text_input(
        "Ask an intelligent question about the PDF:",
        placeholder="e.g., Explain the main concepts, What is machine learning?, Summarize key points..."
    )
    
    if question:
        with st.spinner("🤖 Thinking..."):
            if st.session_state.api_key:
                # Use real AI
                answer = ask_groq_ai(
                    question, 
                    st.session_state.pdf_text, 
                    st.session_state.api_key,
                    st.session_state.selected_model
                )
            else:
                # Demo answer (simulated)
                answer = f"""**AI Answer (Demo Mode):**

To get real AI-powered answers:

1. **Get a free API key** from [console.groq.com](https://console.groq.com)
2. **Enter it in the sidebar**
3. **Ask your question again**

**With AI enabled, you would get:**
- Intelligent analysis of your PDF content
- Context-aware answers based on the document
- Educational explanations
- Page references and specific details

**Example question you asked:** "{question}"

**Try it with your API key!** ⚡"""
            
            # Store in history
            st.session_state.qa_history.append({
                "question": question,
                "answer": answer,
                "model": st.session_state.selected_model if st.session_state.api_key else "Demo",
                "timestamp": "Now"
            })
            
            # Display answer
            st.markdown("---")
            st.markdown("### 🤖 AI Response:")
            st.write(answer)
            
            if st.session_state.api_key:
                st.caption(f"Model: {AVAILABLE_MODELS.get(st.session_state.selected_model, 'AI')}")
    
    # Quick action buttons
    st.markdown("---")
    st.subheader("⚡ Quick Questions")
    
    quick_col1, quick_col2 = st.columns(2)
    
    with quick_col1:
        quick_questions = [
            "What is this document about?",
            "Explain the main concepts",
            "Summarize key points"
        ]
        
        for q in quick_questions:
            if st.button(q, key=f"quick1_{q}", use_container_width=True):
                st.session_state.qa_history.append({
                    "question": q,
                    "answer": f"**[Click to add API key and ask: {q}]**",
                    "model": "Click to enable",
                    "timestamp": "Ready"
                })
                st.rerun()
    
    with quick_col2:
        quick_questions2 = [
            "What are practical applications?",
            "List important topics",
            "Explain methodology"
        ]
        
        for q in quick_questions2:
            if st.button(q, key=f"quick2_{q}", use_container_width=True):
                st.session_state.qa_history.append({
                    "question": q,
                    "answer": f"**[Add API key for AI answer about: {q}]**",
                    "model": "Requires API",
                    "timestamp": "Ready"
                })
                st.rerun()
    
    # Q&A History
    if st.session_state.qa_history:
        st.markdown("---")
        st.subheader("📚 Question History")
        
        for i, qa in enumerate(reversed(st.session_state.qa_history[-5:]), 1):
            with st.expander(f"Q{i}: {qa['question'][:50]}...", expanded=i==1):
                st.write(f"**Question:** {qa['question']}")
                st.write(f"**Answer:** {qa['answer']}")
                if qa.get('model'):
                    st.caption(f"Model: {qa['model']} | {qa.get('timestamp', '')}")
    
    # PDF Preview
    st.markdown("---")
    if st.checkbox("📖 View PDF Text Extract", value=False):
        st.subheader("Extracted PDF Content")
        st.text_area("", st.session_state.pdf_text[:1500] + "..." if len(st.session_state.pdf_text) > 1500 else st.session_state.pdf_text, 
                    height=300, key="pdf_preview")

else:
    # Welcome screen
    st.header("🎯 AI-Powered PDF Education System")
    
    st.write("""
    ### Transform any PDF into an AI tutor!
    
    **Current Features:**
    - ✅ **Latest Groq models** (Llama 3.3, Gemma2, etc.)
    - ✅ **Real AI understanding** of PDF content
    - ✅ **Educational Q&A** with page references
    - ✅ **Multiple model options** in sidebar
    - ✅ **Fast, intelligent responses**
    
    **How to get started:**
    1. **Get free API key** from [console.groq.com](https://console.groq.com) (takes 2 minutes)
    2. **Enter API key** in sidebar
    3. **Select AI model** (Llama 3.3 70B recommended)
    4. **Upload PDF** document
    5. **Ask intelligent questions**
    
    **Perfect for:**
    - 📖 Textbook comprehension
    - 🎓 Research paper analysis
    - 📝 Study guide creation
    - 🔍 Exam preparation
    - 💡 Learning complex concepts
    """)
    
    st.info("""
    **⚡ Why use AI instead of basic search?**
    
    Basic search just finds keywords. AI actually:
    - **Understands context** and meaning
    - **Synthesizes information** from multiple pages
    - **Explains concepts** in simple terms
    - **Answers follow-up questions** intelligently
    - **Provides educational value**
    """)

# Footer
st.markdown("---")
st.write("🤖 **AI-Powered PDF QA System** | Powered by Groq Cloud")
st.caption(f"Using model: {AVAILABLE_MODELS.get(st.session_state.selected_model, 'Select in sidebar')} | Get API key: console.groq.com")
