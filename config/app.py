import streamlit as st
import PyPDF2
import io
from groq import Groq

# ========================================================
# SYSTEM API KEY (Pre-configured by admin)
# ========================================================
try:
    # Load API key from secrets.toml (pre-configured by you)
    API_KEY = st.secrets["GROQ_API_KEY"]
    
    # Initialize Groq client
    groq_client = Groq(api_key=API_KEY)
    AI_ENABLED = True
    
    # Verify API works with a simple test
    try:
        # Quick silent test
        test_response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        st.success("✅ System AI: Ready")
    except Exception as e:
        st.error(f"❌ System AI: Configuration Issue")
        groq_client = None
        AI_ENABLED = False
        
except Exception as e:
    groq_client = None
    AI_ENABLED = False
    st.error(f"⚠️ System AI: Not Available")

# Rest of your code continues...

# Page configuration
st.set_page_config(
    page_title="🤖 StudyMate AI - Professional",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rest of your code continues as is...
# Custom CSS for professional look
st.markdown("""
<style>
    /* Professional Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    /* Answer Display */
    .answer-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e6f7ff 100%);
        padding: 1.8rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 2px solid #3b82f6;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.12);
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Status Indicators */
    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-active {
        background: #10b981;
        color: white;
    }
    
    .status-basic {
        background: #6b7280;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Header Section
st.markdown("""
<div class="main-header">
    <h1 style="margin-bottom: 0.5rem;">🤖 StudyMate AI</h1>
    <h3 style="font-weight: 300; margin-top: 0;">AI-Powered PDF Learning Assistant</h3>
</div>
""", unsafe_allow_html=True)

# Features Section
st.markdown("### 🎯 Professional Features")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="feature-card"><strong>✅ Latest AI Models</strong><br>Llama 3.3, Gemma2</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="feature-card"><strong>✅ Intelligent Q&A</strong><br>Context-aware answers</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="feature-card"><strong>✅ Page Citations</strong><br>Source verification</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="feature-card"><strong>✅ Fast Responses</strong><br>2-3 second answers</div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Status
    if AI_ENABLED:
        st.success("✅ API Key Configured")
        st.caption("🤖 AI Mode: **Active**")
        st.markdown(f'<span class="status-badge status-active">AI ACTIVE</span>', unsafe_allow_html=True)
    else:
        st.error("❌ API Key Issue")
        st.caption("⚠️ Running in Basic Mode")
        st.markdown(f'<span class="status-badge status-basic">BASIC MODE</span>', unsafe_allow_html=True)
    
    st.divider()
    
    st.header("🤖 AI Model")
    model_options = {
        "Llama 3.3 70B (Recommended)": "llama-3.3-70b-versatile",
        "Llama 3.1 8B": "llama-3.1-8b-instant", 
        "Llama 3 70B": "llama3-70b-8192",
        "Mixtral 8x7B": "mixtral-8x7b-32768",
        "Gemma2 9B": "gemma2-9b-it"
    }
    
    selected_model_name = st.selectbox(
        "Choose AI Model",
        list(model_options.keys()),
        index=0
    )
    selected_model = model_options[selected_model_name]
    
    st.caption(f"Model: `{selected_model}`")
    
    st.divider()
    
    st.header("📁 Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Choose PDF file",
        type=["pdf"],
        help="Upload textbooks, research papers, lecture notes"
    )
    
    if uploaded_file:
        file_size = uploaded_file.size / (1024 * 1024)
        st.info(f"**Selected:** {uploaded_file.name} ({file_size:.1f} MB)")
        
        if st.button("🚀 Process PDF with AI", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing document with AI..."):
                try:
                    # Process PDF
                    pdf_bytes = uploaded_file.getvalue()
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                    
                    # Extract all pages
                    pages = []
                    full_text = ""
                    
                    for i in range(len(pdf_reader.pages)):
                        page_text = pdf_reader.pages[i].extract_text()
                        pages.append({
                            'page_num': i + 1,
                            'text': page_text
                        })
                        full_text += f"\n\n--- Page {i+1} ---\n{page_text}"
                    
                    # Store in session state
                    st.session_state.pdf_data = {
                        'filename': uploaded_file.name,
                        'pages': pages,
                        'page_count': len(pdf_reader.pages),
                        'full_text': full_text
                    }
                    
                    st.success(f"✅ PDF processed! ({len(pdf_reader.pages)} pages)")
                    st.info("You can now ask AI-powered questions!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Processing error: {e}")
    
    st.divider()
    
    # System Status
    if AI_ENABLED:
        st.success("🟢 AI System: Ready")
        st.caption("Powered by Groq API")
    else:
        st.warning("⚠️ Basic Mode Active")
        st.caption("Text search only")

# AI Answer Function
def get_ai_answer(question, context, client, model):
    """Get intelligent answer using Groq AI"""
    try:
        prompt = f"""You are StudyMate AI, an expert educational assistant. Answer the student's question based on the provided PDF content.

PDF CONTENT:
{context[:3000]}

STUDENT'S QUESTION: {question}

INSTRUCTIONS:
1. Answer clearly and directly using ONLY the provided PDF content
2. If information exists in the PDF, cite the specific page number(s)
3. If the answer isn't in the PDF, say so honestly
4. Use **bold** for key terms and concepts
5. Format with paragraphs for readability
6. Be educational, helpful, and encouraging

EDUCATIONAL ANSWER:"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert tutor who explains concepts simply and clearly for students."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"**AI Service Note:** Could not generate AI answer. Error: {str(e)}"

# Main content area
if st.session_state.pdf_data:
    # Document info
    st.markdown(f"### 📚 Currently Analyzing: **{st.session_state.pdf_data['filename']}**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pages", st.session_state.pdf_data['page_count'])
    with col2:
        st.metric("AI Model", selected_model_name.split()[0])
    with col3:
        status_text = "✅ AI Active" if AI_ENABLED else "⚠️ Basic Mode"
        st.metric("AI Status", status_text)
    with col4:
        st.metric("Q&A History", len(st.session_state.qa_history))
    
    st.markdown("---")
    
    # Question answering section
    st.markdown("### 💬 Ask AI-Powered Questions")
    
    question = st.text_area(
        "Type your question about the PDF:",
        placeholder="Example: 'Explain the main concepts in this PDF', 'Summarize chapter 2', 'What is machine learning according to this document?', 'Give examples from page 5'",
        height=100
    )
    
    col1, col2 = st.columns([4, 1])
    with col1:
        pass
    with col2:
        ask_button = st.button("🤖 Ask AI", type="primary", use_container_width=True)
    
    if question and ask_button:
        with st.spinner("🧠 Thinking..."):
            # Find relevant pages
            question_lower = question.lower()
            relevant_pages = []
            
            for page in st.session_state.pdf_data['pages']:
                if question_lower in page['text'].lower():
                    relevant_pages.append(page)
            
            # Prepare context for AI
            if relevant_pages:
                context = "\n\n".join([f"Page {p['page_num']}: {p['text'][:500]}" for p in relevant_pages[:3]])
                source_pages = [p['page_num'] for p in relevant_pages[:3]]
            else:
                # Use first few pages if no direct match
                context = "\n\n".join([f"Page {p['page_num']}: {p['text'][:300]}" for p in st.session_state.pdf_data['pages'][:2]])
                source_pages = [1, 2]
            
            # Get answer
            if AI_ENABLED and groq_client:
                answer = get_ai_answer(question, context, groq_client, selected_model)
                answer_type = "🤖 **AI-Powered Answer**"
            else:
                # Basic text-based answer
                answer = f"**Basic Answer from Document:**\n\n{context[:500]}..."
                answer_type = "📄 **Text-Based Answer**"
            
            # Store in history
            st.session_state.qa_history.append({
                'question': question,
                'answer': answer,
                'source_pages': source_pages,
                'ai_used': AI_ENABLED and bool(groq_client),
                'timestamp': len(st.session_state.qa_history)
            })
            
            # Display answer
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(answer_type)
            st.write(answer)
            if source_pages:
                st.caption(f"📄 **Source Pages:** {', '.join(map(str, source_pages))}")
            if AI_ENABLED and groq_client:
                st.caption(f"🤖 **AI Model:** {selected_model_name}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Suggested questions
    st.markdown("---")
    st.markdown("### 🎓 Educational Question Examples")
    
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        examples = [
            "Define key terms from this PDF",
            "Explain the main concepts",
            "Summarize the document",
            "What are the applications?"
        ]
        
        for ex in examples:
            if st.button(ex, key=f"ex1_{ex}", use_container_width=True):
                st.session_state.qa_history.append({
                    'question': ex,
                    'answer': f"Sample answer for: {ex}\n\nTry asking this question to see the AI response!",
                    'source_pages': [1],
                    'ai_used': False,
                    'timestamp': len(st.session_state.qa_history)
                })
                st.rerun()
    
    with example_col2:
        examples2 = [
            "Give examples from the PDF",
            "What problems are addressed?",
            "Compare different approaches",
            "List key findings"
        ]
        
        for ex in examples2:
            if st.button(ex, key=f"ex2_{ex}", use_container_width=True):
                st.session_state.qa_history.append({
                    'question': ex,
                    'answer': f"Sample answer for: {ex}\n\nTry asking this question to see the AI response!",
                    'source_pages': [1],
                    'ai_used': False,
                    'timestamp': len(st.session_state.qa_history)
                })
                st.rerun()
    
    # Q&A History
    if st.session_state.qa_history:
        st.markdown("---")
        st.markdown("### 📝 Learning History")
        
        for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):
            with st.expander(f"Q{len(st.session_state.qa_history)-i}: {qa['question'][:50]}...", expanded=(i==0)):
                st.markdown(f"**Question:** {qa['question']}")
                st.markdown("---")
                st.markdown(f"**Answer:** {qa['answer']}")
                if qa.get('source_pages'):
                    st.caption(f"📄 Source Pages: {', '.join(map(str, qa['source_pages']))}")
                st.caption(f"🤖 {'AI-Powered' if qa.get('ai_used') else 'Text Search'}")
    
    # Document preview
    st.markdown("---")
    if st.checkbox("📖 Explore Document Content", value=False):
        st.markdown("### 🔍 Document Preview")
        
        page_num = st.slider(
            "Select page to preview:", 
            1, 
            st.session_state.pdf_data['page_count'], 
            1
        )
        
        page_text = st.session_state.pdf_data['pages'][page_num-1]['text']
        
        st.text_area(
            f"Page {page_num} content:",
            page_text[:1500] + "..." if len(page_text) > 1500 else page_text,
            height=200,
            disabled=True
        )
        
        st.caption(f"📊 Page {page_num} has approximately {len(page_text.split())} words")

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: #f8fafc; border-radius: 15px;'>
        <h2 style='color: #4f46e5;'>🎓 Welcome to StudyMate AI</h2>
        <p style='font-size: 1.1rem; color: #64748b; margin-top: 1rem; margin-bottom: 2rem;'>
        Your professional AI learning assistant is ready!
        </p>
        
        <div style='display: inline-block; background: white; padding: 1rem 2rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
            <div style='font-size: 1.2rem; color: #10b981; font-weight: bold;'>
                ✅ API Key: {"CONFIGURED" if AI_ENABLED else "MISSING"}
            </div>
            <div style='font-size: 0.9rem; color: #6b7280; margin-top: 0.5rem;'>
                {"AI System Active" if AI_ENABLED else "Basic Mode Only"}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📚 **How to Get Started**
        
        **Step-by-Step:**
        1. **Upload** your textbook/research PDF using the sidebar
        2. **Process** the document (click "Process PDF with AI")
        3. **Ask Questions** about any concept in the PDF
        4. **Learn** from AI-powered explanations with citations
        
        **Perfect For:**
        - **Students**: Understand complex textbooks faster
        - **Researchers**: Analyze papers efficiently  
        - **Teachers**: Create interactive learning materials
        - **Professionals**: Review technical documents
        
        **AI Features:**
        • Intelligent concept explanations
        • Page-specific citations
        • Learning history tracking
        • Multiple AI model options
        """)
    
    with col2:
        st.info("""
        **💡 Pro Tips:**
        
        1. **Be specific** - Ask detailed questions
        2. **Use page references** - "Explain from page 15"
        3. **Ask follow-ups** - Build on previous answers
        4. **Check citations** - Verify information sources
        
        **🎯 Example Questions:**
        • "What is supervised learning?"
        • "Summarize chapter 3"
        • "Compare X and Y concepts"
        • "Give examples from the document"
        
        **🚀 Ready when you are!**
        """)

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
with footer_col1:
    if AI_ENABLED:
        st.markdown("**🤖 StudyMate AI Professional** • 🟢 AI System Active")
    else:
        st.markdown("**🤖 StudyMate AI** • ⚪ Basic Mode")
with footer_col2:
    st.caption(f"Model: {selected_model_name}")
with footer_col3:
    status = "🟢 AI Active" if AI_ENABLED else "⚪ Basic Mode"
    st.caption(f"Status: {status}")