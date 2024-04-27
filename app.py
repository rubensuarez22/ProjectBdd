import streamlit as st
import pdfplumber
from collections import Counter, defaultdict
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd


# Ensure NLTK components are downloaded
nltk.download('punkt')
nltk.download('stopwords')

st.title("Advanced Document Management System")

# Sidebar navigation setup
option = st.sidebar.selectbox(
    'Choose a function',
    ('Upload Documents', 'Frequency Matrix', 'Indexing Terms', 'Document Query')
)

def process_text(text):
    """Tokenizes, removes stopwords, and stems the text."""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens if word.isalpha() and word.lower() not in stop_words]

if option == "Upload Documents":
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type='pdf')
    if uploaded_files:
        document_texts = {}
        for uploaded_file in uploaded_files:
            with pdfplumber.open(uploaded_file) as pdf:
                full_text = ''
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text
                document_texts[uploaded_file.name] = full_text
        st.session_state['documents'] = document_texts
        st.success("Files uploaded and text extracted successfully!")

elif option == "Frequency Matrix":
    st.header("Frequency Matrix")
    if 'documents' in st.session_state and st.session_state['documents']:
        # Table of document identifiers and titles
        doc_data = [{"Identifier": f"d{i+1}", "Article Title": text.split('\n')[0]} for i, text in enumerate(st.session_state['documents'].values())]
        df_docs = pd.DataFrame(doc_data)
        st.table(df_docs)
        
        # Process each document for word frequencies
        doc_frequencies = defaultdict(dict)
        doc_names = list(st.session_state['documents'].keys())
        for doc_name, text in st.session_state['documents'].items():
            processed_text = process_text(text)
            word_count = Counter(processed_text)
            for word, count in word_count.items():
                doc_frequencies[word][doc_name] = count

        # Create DataFrame from the nested dictionary
        df_freq = pd.DataFrame.from_dict(doc_frequencies, orient='index').fillna(0).astype(int)
        # Rename columns to identifiers like d1, d2, etc.
        doc_ids = {doc_name: f"d{i+1}" for i, doc_name in enumerate(doc_names)}
        df_freq.rename(columns=doc_ids, inplace=True)

        st.write("Term-Document Frequency Matrix:")
        st.dataframe(df_freq)
        st.session_state['df_freq'] = df_freq

    else:
        st.error("Please upload documents first using the 'Upload Documents' section.")

# ... [The previous parts of your script remain unchanged]

elif option == "Indexing Terms":
    st.header("Indexing Terms")
    st.subheader("Decompose the frequency matrix using SVD")

    num_terms_to_retain = st.number_input('Enter the number of terms to retain (K):', min_value=1, max_value=None, value=200)

    if st.button('Perform SVD'):
        if 'df_freq' in st.session_state:
            df_freq = st.session_state['df_freq']  # Retrieve the frequency matrix

            # Ensure terms are mapped correctly (if using TfidfVectorizer earlier, otherwise assume df_freq has proper terms)
            if 'vectorizer' in st.session_state:
                terms = st.session_state['vectorizer'].get_feature_names_out()
            else:
                terms = df_freq.index  # Assuming df_freq's index contains the actual term names

            # Perform SVD
            U, Sigma, VT = np.linalg.svd(df_freq, full_matrices=False)

            # Determine the number of terms to retain
            k = min(num_terms_to_retain, len(Sigma))

            # Truncate matrices
            U_k = U[:, :k]
            Sigma_k = Sigma[:k]
            VT_k = VT[:k, :]

            # Display the truncated U matrix with terms
            st.write("Truncated U matrix (Term-Concepts):")
            U_k_df = pd.DataFrame(U_k, index=terms[:len(U)], columns=[f'Component_{i}' for i in range(k)])
            st.dataframe(U_k_df)

            # Display the singular values (Sigma) alongside the terms
            st.write("Truncated Sigma (Singular Values):")
            Sigma_k_df = pd.DataFrame({'Singular Value': Sigma_k}, index=terms[:k])
            st.dataframe(Sigma_k_df)

            # Display the truncated VT matrix
            st.write("Truncated VT matrix (Concept-Documents):")
            VT_k_df = pd.DataFrame(VT_k, columns=df_freq.columns)
            st.dataframe(VT_k_df)

            # Explained variance
            st.write("Explained variance by retained components:")
            total_variance = np.sum(Sigma**2)
            explained_variance = [(s**2 / total_variance) for s in Sigma_k]
            st.bar_chart(explained_variance)
        else:
            st.error("Please calculate the frequency matrix first.")





elif option == "Document Query":
    st.header("Document Query using SQL")
    # Placeholder for actual querying functionality
    st.subheader("Execute queries to find document similarities and relevancies")

elif option == "Document Query":
    st.header("Document Query using SQL")
    st.subheader("Execute queries to find document similarities and relevancies")
    # Inputs for SQL-based document querying
    st.write("Query the document base to evaluate similarities or fetch the most relevant documents based on your query.")
    
    st.write("### Document Similarity")
    doc1 = st.text_input("Enter the first document name/id:", key="doc1")
    doc2 = st.text_input("Enter the second document name/id:", key="doc2")
    if st.button("Evaluate Similarity between Documents"):
        st.write(f"Similarity result between documents {doc1} and {doc2} would be calculated and shown here.")

    st.write("### Document Relevancy")
    query = st.text_input("Enter your query:", key="query")
    n_docs = st.slider("Select number of documents", 1, 20, 5, key="n_docs")
    if st.button("Fetch Documents Based on Query"):
        st.write(f"Fetching {n_docs} most relevant documents for the query: `{query}`")

# The st.button will rerun the script from the top when clicked, so state management might be necessary for more complex interactions.
