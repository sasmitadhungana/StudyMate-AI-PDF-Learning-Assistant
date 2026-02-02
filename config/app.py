

import streamlit as st
import PyPDF2
import io

st.set_page_config(
    page_title="PDF QA Education System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .answer-box {
        background-color: #f0f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin: 1rem 0;
    }
    .question-box {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 PDF QA Education System")
st.markdown("**Extract knowledge from your educational PDFs**")
st.markdown("---")

# Initialize session state
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Enhanced answer function
def get_answer_from_pdf(question, pdf_data):
    """Get intelligent answer from PDF content"""
    if not pdf_data:
        return "Please upload and process a PDF first.", None
    
    question_lower = question.lower()
    pages = pdf_data['pages']
    full_text = pdf_data['full_text'].lower()
    
    # Check for specific question types
    if any(word in question_lower for word in ['what is', 'define', 'definition']):
        # Look for definitions
        for i, page in enumerate(pages):
            if question_lower in page.lower():
                # Find the sentence containing the answer
                sentences = page.split('.')
                for sentence in sentences:
                    if question_lower in sentence.lower():
                        return f"**Definition from Page {i+1}:**\n\n{sentence.strip()}.", i+1
    
    elif 'summar' in question_lower:
        # Provide summary
        summary_parts = []
        for i, page in enumerate(pages[:2]):  # First 2 pages
            if len(page) > 100:
                summary_parts.append(f"**Page {i+1}:** {page[:200]}...")
        return "**Document Summary:**\n\n" + "\n\n".join(summary_parts), None
    
    elif any(word in question_lower for word in ['example', 'example of', 'for example']):
        # Look for examples
        for i, page in enumerate(pages):
            if 'example' in page.lower():
                lines = page.split('\n')
                for line in lines:
                    if 'example' in line.lower() and len(line) > 20:
                        return f"**Example from Page {i+1}:**\n\n{line}", i+1
    
    # General search - find best matching page
    best_page = 0
    best_score = 0
    
    for i, page in enumerate(pages):
        page_lower = page.lower()
        score = sum(1 for word in question_lower.split() 
                   if len(word) > 3 and word in page_lower)
        
        if score > best_score:
            best_score = score
            best_page = i
    
    if best_score > 0 and best_page < len(pages):
        # Extract context from the best page
        page_text = pages[best_page]
        # Find a relevant sentence
        sentences = page_text.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in question_lower.split() if len(word) > 3):
                return f"**From Page {best_page + 1}:**\n\n{sentence.strip()}.", best_page + 1
        
        # Fallback: first part of the page
        return f"**Relevant content from Page {best_page + 1}:**\n\n{page_text[:300]}...", best_page + 1
    
    return "I couldn't find specific information about that in the PDF. Try asking about general topics or key terms from the document.", None

# Sidebar for PDF processing
with st.sidebar:
    st.header("📁 Document Management")
    
    uploaded_file = st.file_uploader(
        "Upload Educational PDF",
        type=["pdf"],
        help="Upload textbooks, research papers, lecture notes"
    )
    
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")
        
        if st.button("🔬 Process PDF for Q&A", type="primary", use_container_width=True):
            with st.spinner("Analyzing document content..."):
                try:
                    # Process PDF
                    pdf_bytes = uploaded_file.getvalue()
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                    
                    # Extract all pages
                    pages = []
                    full_text = ""
                    
                    for i in range(len(pdf_reader.pages)):
                        page_text = pdf_reader.pages[i].extract_text()
                        pages.append(page_text)
                        full_text += f"\n\n--- Page {i+1} ---\n{page_text}"
                    
                    # Store in session state
                    st.session_state.pdf_data = {
                        'filename': uploaded_file.name,
                        'pages': pages,
                        'page_count': len(pdf_reader.pages),
                        'full_text': full_text
                    }
                    
                    st.success(f"📚 Document ready! ({len(pdf_reader.pages)} pages)")
                    st.info("You can now ask questions about this PDF")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Processing error: {e}")
    
    st.divider()
    
    # Quick stats
    if st.session_state.pdf_data:
        st.header("📊 Document Info")
        st.metric("Total Pages", st.session_state.pdf_data['page_count'])
        st.metric("Status", "Ready for Q&A")
        st.success("✅ Document analyzed")

# Main content area
if st.session_state.pdf_data:
    # Document info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Document", st.session_state.pdf_data['filename'])
    with col2:
        st.metric("Pages", st.session_state.pdf_data['page_count'])
    with col3:
        st.metric("Q&A Ready", "✅")
    
    st.markdown("---")
    
    # Question answering section
    st.header("💬 Educational Q&A")
    
    # Question input
    question = st.text_input(
        "Ask a question about the document:",
        placeholder="e.g., Explain supervised learning, What are the key concepts?, Give an example of..."
    )
    
    if question:
        # Get answer
        with st.spinner("🔍 Searching document for answer..."):
            answer, page_num = get_answer_from_pdf(question, st.session_state.pdf_data)
            
            # Store in history
            st.session_state.qa_history.append({
                'question': question,
                'answer': answer,
                'page': page_num
            })
            
            # Display answer
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown("### 🤖 Answer from Document:")
            st.write(answer)
            if page_num:
                st.caption(f"📄 Source: Page {page_num}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Suggested questions for education
    st.markdown("---")
    st.subheader("🎓 Educational Question Types:")
    
    edu_col1, edu_col2 = st.columns(2)
    
    with edu_col1:
        edu_questions = [
            "Define key terms",
            "Explain main concepts",
            "What is the methodology?",
            "List examples provided"
        ]
        
        for q in edu_questions:
            if st.button(q, key=f"edu1_{q}", use_container_width=True):
                st.session_state.qa_history.append({
                    'question': q,
                    'answer': get_answer_from_pdf(q, st.session_state.pdf_data)[0],
                    'page': get_answer_from_pdf(q, st.session_state.pdf_data)[1]
                })
                st.rerun()
    
    with edu_col2:
        edu_questions2 = [
            "Summarize the document",
            "What are applications?",
            "Compare different approaches",
            "What problems are addressed?"
        ]
        
        for q in edu_questions2:
            if st.button(q, key=f"edu2_{q}", use_container_width=True):
                st.session_state.qa_history.append({
                    'question': q,
                    'answer': get_answer_from_pdf(q, st.session_state.pdf_data)[0],
                    'page': get_answer_from_pdf(q, st.session_state.pdf_data)[1]
                })
                st.rerun()
    
    # Q&A History
    if st.session_state.qa_history:
        st.markdown("---")
        st.subheader("📝 Learning History")
        
        for i, qa in enumerate(reversed(st.session_state.qa_history[-5:]), 1):
            with st.expander(f"Q{i}: {qa['question'][:40]}...", expanded=i==1):
                st.markdown('<div class="question-box">', unsafe_allow_html=True)
                st.write(f"**Question:** {qa['question']}")
                st.markdown('</div>', unsafe_allow_html=True)
                st.write(f"**Answer:** {qa['answer']}")
                if qa.get('page'):
                    st.caption(f"📖 Source: Page {qa['page']}")
    
    # Document preview
    st.markdown("---")
    if st.checkbox("📖 View Document Preview", value=False):
        st.subheader("Document Content Preview")
        if st.session_state.pdf_data['pages']:
            preview_page = st.slider("Select page to preview", 1, 
                                   st.session_state.pdf_data['page_count'], 1)
            page_text = st.session_state.pdf_data['pages'][preview_page-1]
            st.text_area(f"Page {preview_page} content:", 
                        page_text[:1000] + "..." if len(page_text) > 1000 else page_text,
                        height=200)

else:
    # Welcome screen
    st.header("🎯 Welcome to PDF QA Education System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        ### Transform your educational PDFs into interactive learning tools!
        
        **How it works:**
        1. **Upload** any educational PDF (textbooks, papers, notes)
        2. **Process** the document to extract all text content
        3. **Ask questions** about the material
        4. **Get answers** directly from the document
        
        **Perfect for:**
        - 📖 Textbook comprehension
        - 🎓 Research paper analysis  
        - 📝 Lecture note review
        - 🔍 Exam preparation
        
        **Try it now:** Upload a PDF using the sidebar
        """)
    
    with col2:
        st.info("""
        **Example Questions:**
        
        • "What is [concept]?"
        • "Explain [topic]"
        • "Summarize chapter 1"
        • "List key points"
        • "Give examples of..."
        
        *Based on your uploaded PDF*
        """)

# Footer
st.markdown("---")
st.write("📚 **PDF QA Education System** | Powered by document analysis")
st.caption("Upload → Analyze → Question → Learn")
