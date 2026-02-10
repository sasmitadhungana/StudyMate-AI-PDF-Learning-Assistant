import streamlit as st
import PyPDF2
import io
from groq import Groq

# ========================================================
# SYSTEM API KEY (Admin Pre-configured)
# ========================================================
try:
    # Load API key from secrets.toml
    API_KEY = st.secrets["GROQ_API_KEY"]
    
    if API_KEY and len(API_KEY) > 20:
        groq_client = Groq(api_key=API_KEY)
        AI_ENABLED = True
    else:
        groq_client = None
        AI_ENABLED = False
        
except Exception as e:
    groq_client = None
    AI_ENABLED = False

# Page configuration
st.set_page_config(
    page_title="StudyMate AI | PDF Learning Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS - Clean and Minimal
st.markdown("""
<style>
    /* Clean Professional Header */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(30, 60, 114, 0.15);
    }
    
    .system-status {
        background: rgba(255,255,255,0.15);
        padding: 12px 20px;
        border-radius: 8px;
        margin: 20px auto;
        width: fit-content;
        font-size: 0.9rem;
        border-left: 4px solid #4CAF50;
    }
    
    /* Clean Cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Professional Answer Box */
    .answer-box {
        background: #f8fafc;
        padding: 1.8rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        border-left: 4px solid #3b82f6;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    /* Clean Buttons */
    .stButton button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.95rem;
        width: 100%;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.2);
    }
    
    /* Subtle Metrics */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    
    /* Clean Sidebar */
    .sidebar-header {
        color: #1e3c72;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Professional typography */
    h1, h2, h3 {
        font-weight: 600;
        color: #1e293b;
    }
    
    .subtitle {
        color: #64748b;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# ========================================================
# CLEAN PROFESSIONAL HEADER
# ========================================================
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("""
    <div class="main-header">
        <h1 style="margin-bottom: 0.8rem;">📚 StudyMate AI</h1>
        <p class="subtitle" style="font-size: 1.1rem; margin-bottom: 1rem;">
            Intelligent PDF Learning Assistant
        </p>
        <div class="system-status">
            <span style="color: #4CAF50;">●</span> AI Assistant Ready
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========================================================
# CLEAN SIDEBAR - NO API MENTIONS
# ========================================================
with st.sidebar:
    # System Status
    st.markdown('<div class="sidebar-header">SYSTEM STATUS</div>', unsafe_allow_html=True)
    
    if AI_ENABLED:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("🟢")
        with col2:
            st.markdown("**AI Assistant**<br><span style='font-size: 0.85rem; color: #666;'>Online</span>", unsafe_allow_html=True)
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("⚪")
        with col2:
            st.markdown("**Basic Mode**<br><span style='font-size: 0.85rem; color: #666;'>Text Search</span>", unsafe_allow_html=True)
    
    st.divider()
    
    # AI Model Selection
    st.markdown('<div class="sidebar-header">AI MODEL</div>', unsafe_allow_html=True)
    model_options = {
        "Llama 3.3 70B": "llama-3.3-70b-versatile",
        "Llama 3.1 8B": "llama-3.1-8b-instant", 
        "Gemma2 9B": "gemma2-9b-it",
        "Mixtral 8x7B": "mixtral-8x7b-32768"
    }
    
    selected_model = st.selectbox(
        "",
        list(model_options.keys()),
        index=0,
        label_visibility="collapsed"
    )
    model_id = model_options[selected_model]
    
    st.caption(f"Model: `{model_id.split('-')[0]}`")
    
    st.divider()
    
    # Document Upload
    st.markdown('<div class="sidebar-header">DOCUMENT UPLOAD</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "",
        type=["pdf"],
        help="Upload textbooks, research papers, or lecture notes",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        file_size = uploaded_file.size / (1024 * 1024)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{uploaded_file.name}**<br><span style='font-size: 0.85rem; color: #666;'>{file_size:.1f} MB</span>", unsafe_allow_html=True)
        with col2:
            if st.button("📄", help="Process document"):
                with st.spinner("Processing..."):
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
                        
                        st.success(f"✓ {len(pdf_reader.pages)} pages loaded")
                        st.rerun()
                        
                    except Exception as e:
                        st.error("Document processing failed")
    
    st.divider()
    
    # Quick Actions
    st.markdown('<div class="sidebar-header">QUICK ACTIONS</div>', unsafe_allow_html=True)
    
    if st.session_state.pdf_data:
        if st.button("🗑️ Clear Document", use_container_width=True):
            st.session_state.pdf_data = None
            st.rerun()
    
    if st.session_state.qa_history:
        if st.button("📊 Export History", use_container_width=True):
            # Create export data
            export_text = "StudyMate AI - Learning History\n"
            export_text += "=" * 50 + "\n\n"
            
            for i, qa in enumerate(st.session_state.qa_history):
                export_text += f"Q{i+1}: {qa['question']}\n"
                export_text += f"A: {qa['answer'][:200]}...\n"
                export_text += "-" * 40 + "\n"
            
            st.download_button(
                label="Download",
                data=export_text,
                file_name="learning_history.txt",
                mime="text/plain"
            )

# ========================================================
# MAIN CONTENT AREA
# ========================================================
if not st.session_state.pdf_data:
    # Clean Welcome Screen
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Welcome to StudyMate AI")
        st.markdown("""
        Transform any document into an interactive learning experience. 
        Upload PDFs to access intelligent Q&A, summaries, and insights.
        """)
        
        # Feature Cards
        st.markdown("#### Core Features")
        
        features = [
            ("🤖", "AI-Powered Analysis", "Deep understanding of document content"),
            ("📄", "Page References", "Answers linked to specific pages"),
            ("💬", "Interactive Q&A", "Ask questions in natural language"),
            ("⚡", "Fast Processing", "Instant responses with latest AI models")
        ]
        
        cols = st.columns(2)
        for i, (icon, title, desc) in enumerate(features):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="feature-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-weight: 600; margin-bottom: 0.25rem;">{title}</div>
                    <div style="font-size: 0.9rem; color: #666;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Getting Started")
        st.markdown("""
        1. **Upload** a PDF document using the sidebar
        2. **Process** the document with one click
        3. **Ask** questions about the content
        4. **Learn** with AI-powered insights
        """)
        
        st.markdown("---")
        
        st.markdown("#### Supported Content")
        st.markdown("""
        - Textbooks & Study Guides
        - Research Papers
        - Lecture Notes
        - Technical Documents
        - Reports & Whitepapers
        """)

else:
    # Document Analysis Interface
    st.markdown(f"### 📑 {st.session_state.pdf_data['filename']}")
    
    # Document Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div style="font-size: 0.9rem; color: #666;">Pages</div><div style="font-size: 1.5rem; font-weight: 600;">{}</div></div>'.format(st.session_state.pdf_data['page_count']), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div style="font-size: 0.9rem; color: #666;">AI Model</div><div style="font-size: 1.2rem; font-weight: 600;">{}</div></div>'.format(selected_model.split()[0]), unsafe_allow_html=True)
    with col3:
        status = "AI Active" if AI_ENABLED else "Text Search"
        st.markdown('<div class="metric-card"><div style="font-size: 0.9rem; color: #666;">Mode</div><div style="font-size: 1.2rem; font-weight: 600;">{}</div></div>'.format(status), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div style="font-size: 0.9rem; color: #666;">Questions</div><div style="font-size: 1.5rem; font-weight: 600;">{}</div></div>'.format(len(st.session_state.qa_history)), unsafe_allow_html=True)
    
    st.divider()
    
    # Q&A Section
    st.markdown("### Ask Questions")
    
    question = st.text_area(
        "Enter your question about the document:",
        placeholder="e.g., 'Explain the main concepts', 'Summarize key findings', 'What is discussed on page 5?'",
        height=100,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([6, 1])
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    
    # AI Answer Function
    def get_ai_answer(question, context, client, model):
        """Get intelligent answer using Groq AI"""
        try:
            prompt = f"""You are an expert educational assistant. Answer the question based on the provided document content.

DOCUMENT CONTENT:
{context[:3000]}

QUESTION: {question}

INSTRUCTIONS:
1. Answer clearly using ONLY the provided document content
2. Cite specific page numbers when relevant
3. If information isn't in the document, state this honestly
4. Use clear, professional language
5. Focus on educational value

ANSWER:"""
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable tutor providing clear, accurate information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"**Note:** Unable to generate AI response. Please try again."
    
    if question and ask_button:
        with st.spinner("Analyzing..."):
            # Find relevant pages
            question_lower = question.lower()
            relevant_pages = []
            
            for page in st.session_state.pdf_data['pages']:
                if question_lower in page['text'].lower():
                    relevant_pages.append(page)
            
            # Prepare context
            if relevant_pages:
                context = "\n\n".join([f"Page {p['page_num']}: {p['text'][:500]}" for p in relevant_pages[:3]])
                source_pages = [p['page_num'] for p in relevant_pages[:3]]
            else:
                context = "\n\n".join([f"Page {p['page_num']}: {p['text'][:300]}" for p in st.session_state.pdf_data['pages'][:2]])
                source_pages = [1, 2]
            
            # Get answer
            if AI_ENABLED and groq_client:
                answer = get_ai_answer(question, context, groq_client, model_id)
                answer_type = "AI Analysis"
            else:
                answer = f"**Document Content:**\n\n{context[:500]}..."
                answer_type = "Text Search"
            
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
            st.markdown(f"**{answer_type}**")
            st.write(answer)
            if source_pages:
                st.caption(f"📑 Source pages: {', '.join(map(str, source_pages))}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Suggested Questions
    if len(st.session_state.qa_history) == 0:
        st.markdown("#### Try These Questions")
        
        suggestions = [
            "What are the main topics covered?",
            "Summarize the key points",
            "Explain important concepts",
            "What problems are addressed?"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, use_container_width=True):
                    st.session_state.last_question = suggestion
                    st.rerun()
    
    # Learning History
    if st.session_state.qa_history:
        st.divider()
        st.markdown("#### Learning History")
        
        for i, qa in enumerate(reversed(st.session_state.qa_history[-3:])):
            with st.expander(f"Q{len(st.session_state.qa_history)-i}: {qa['question'][:60]}...", expanded=(i==0)):
                st.markdown(f"**Question:** {qa['question']}")
                st.markdown("---")
                st.markdown(f"**Answer:** {qa['answer']}")
                if qa.get('source_pages'):
                    st.caption(f"📑 Pages: {', '.join(map(str, qa['source_pages']))}")

# ========================================================
# CLEAN FOOTER
# ========================================================
st.divider()
footer_col1, footer_col2 = st.columns([3, 1])
with footer_col1:
    st.markdown("<span style='font-size: 0.9rem; color: #666;'>StudyMate AI • Intelligent Document Analysis</span>", unsafe_allow_html=True)
with footer_col2:
    status_indicator = "🟢 AI" if AI_ENABLED else "⚪ Basic"
    st.markdown(f"<span style='font-size: 0.9rem; color: #666;'>{status_indicator}</span>", unsafe_allow_html=True)