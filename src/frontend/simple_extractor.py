import streamlit as st
import PyPDF2
import io

st.title("PDF TEXT EXTRACTOR")
st.write("Upload PDF -> Click Process -> See Text")

# Upload
pdf_file = st.file_uploader("Choose PDF", type=["pdf"])

if pdf_file:
    st.write(f"File: {pdf_file.name}")
    
    # Process button
    if st.button("EXTRACT TEXT NOW", type="primary"):
        # Show processing
        with st.spinner("Extracting text..."):
            try:
                # Read PDF
                pdf_bytes = pdf_file.getvalue()
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                
                # Show pages
                st.success(f"✅ {len(pdf_reader.pages)} pages found")
                
                # Extract and show text
                for i in range(min(3, len(pdf_reader.pages))):  # Show first 3 pages
                    page = pdf_reader.pages[i]
                    text = page.extract_text()
                    
                    with st.expander(f"Page {i+1}", expanded=i==0):
                        st.text(text[:1000] + "..." if len(text) > 1000 else text)
                
                # Ask question
                st.markdown("---")
                question = st.text_input("Ask about this PDF:")
                
                if question:
                    # Simple search
                    found = False
                    for i in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[i]
                        text = page.extract_text()
                        if question.lower() in text.lower():
                            st.info(f"Found on page {i+1}: {text[:200]}...")
                            found = True
                            break
                    
                    if not found:
                        st.warning("Not found. Try different keywords.")
                        
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Upload a PDF file to begin")

st.caption("Working PDF Text Extractor")
