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

    else:
        st.error("Please upload documents first using the 'Upload Documents' section.")

elif option == "Indexing Terms":
    st.header("Indexing Terms")
    st.subheader("Apply SVD to reduce dimensionality and focus on significant terms")
    if 'documents' in st.session_state and st.session_state['documents']:
        # Prepare the text data for SVD
        documents = [text for text in st.session_state['documents'].values()]
        vectorizer = TfidfVectorizer(tokenizer=process_text, stop_words='english')
        X = vectorizer.fit_transform(documents)
        # Perform SVD
        svd_model = TruncatedSVD(n_components=2, random_state=42)
        lsa = svd_model.fit_transform(X)

        np.random.seed(42)
        mock_data = np.random.randint(1, 20, size=(10, 5))  # Replace with your actual frequency matrix
        frequency_matrix = pd.DataFrame(mock_data, columns=[f'Doc {i+1}' for i in range(mock_data.shape[1])],
        index=[f'Term {i+1}' for i in range(mock_data.shape[0])])

        st.write("Frequency Matrix `FrecT`:")
        st.dataframe(frequency_matrix)

        U, s, VT = svd(frequency_matrix.values, full_matrices=False)
        Sigma = np.diag(s)

        # Convert SVD components into DataFrame for display
        U_df = pd.DataFrame(U, index=[f'Term {i+1}' for i in range(U.shape[0])])
        Sigma_df = pd.DataFrame(Sigma)
        VT_df = pd.DataFrame(VT, columns=[f'Doc {i+1}' for i in range(VT.shape[1])])

        # Display SVD components
        st.write("Left Singular Vectors (U matrix):")
        st.dataframe(U_df)

        st.write("Singular Values (Sigma matrix):")
        st.dataframe(Sigma_df)

        st.write("Right Singular Vectors Transposed (`VT` matrix):")
        st.dataframe(VT_df)

        st.write("""
        Given a frequency matrix `FrecT`, it can be decomposed into an SVD `T x S x D^T`, where `S` is non-increasing.
        The dimensions of `T` are MxR, `S` is RxR, and `D^T` is RxN.
        In this decomposition, M is the number of terms, N is the number of documents, and R is the rank of the matrix.
        """)
        # Plot the documents in the concept space
        fig, ax = plt.subplots()
        ax.scatter(lsa[:, 0], lsa[:, 1])
        ax.set_title('Document Clustering Post SVD')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')

        for i, txt in enumerate(st.session_state['documents'].keys()):
            ax.annotate(txt, (lsa[i, 0], lsa[i, 1]))

        st.pyplot(fig)


        # Display the singular values (importance of each component)
        st.write("Singular values:", svd_model.singular_values_)

        # Display the top significant terms per topic/component
        terms = vectorizer.get_feature_names_out()
        for i, comp in enumerate(svd_model.components_):
            terms_comp = zip(terms, comp)
            sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:10]
            st.write("Topic " + str(i+1) + ": ")
            for t in sorted_terms:
                st.write(t[0], round(t[1], 3))



        

    else:
        st.error("Please upload documents first.")

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
