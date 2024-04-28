import streamlit as st
import pdfplumber
from collections import Counter, defaultdict
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from numpy.linalg import svd as svd
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Text

# Ensure NLTK components are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Create our database
engine = create_engine('sqlite:///mydatabase.db')

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    content = Column(Text)

# Create the tables in the database
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

st.title("Advanced Document Management System")

# Sidebar navigation setup
option = st.sidebar.selectbox(
    'Choose a function',
    ('Upload Documents', 'View Documents','Frequency Matrix', 'Indexing Terms', 'Document Query')
)

def tokenize(text):
    """Tokenizes the text without removing any words."""
    return [word for word in word_tokenize(text) if word.isalpha()]

def remove_stopwords(text):
    """Tokenizes the text and removes stopwords."""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.isalpha() and word.lower() not in stop_words]

def apply_stemming(text):
    """Tokenizes and applies stemming to the text."""
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens if word.isalpha()]

def process_text_full(text):
    """Tokenizes, removes stopwords, and applies stemming to the text."""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens if word.isalpha() and word.lower() not in stop_words]

# Processing the documents based on the option
document_processing_functions = {
    "Original": tokenize,
    "Stop List Removed": remove_stopwords,
    "Suffix List Removed": apply_stemming,
    "Word Stems": process_text_full
}

additional_stopwords = {'example', 'another'}

if option == "Upload Documents":
    st.header("Upload Documents")

    if st.button("Clear Database"):
        session.query(Document).delete()
        session.commit()
    
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
            # Insert each document into the database
            new_document = Document(title=uploaded_file.name, content=full_text)
            session.add(new_document)
        session.commit()  # Commit the transaction to the database
        st.session_state['documents'] = document_texts
        st.success("Files uploaded and text extracted successfully!")

# View documents
elif option == "View Documents":
    st.header("View Uploaded Documents")
    documents = session.query(Document).all()
    if documents:
        for doc in documents:
            st.subheader(f"Document ID: {doc.id}, Title: {doc.title}")
    else:
        st.write("No documents found.")

elif option == "Frequency Matrix":
    st.header("Frequency Matrix")
    if 'documents' in st.session_state and st.session_state['documents']:
        matrix_data = {}
        doc_names = list(st.session_state['documents'].keys())

        for matrix_type, processing_function in document_processing_functions.items():
            doc_frequencies = defaultdict(dict)
            for doc_name, text in st.session_state['documents'].items():
                processed_text = processing_function(text)
                word_count = Counter(processed_text)
                for word, count in word_count.items():
                    doc_frequencies[word][doc_name] = count

            # Create DataFrame from the nested dictionary
            df = pd.DataFrame.from_dict(doc_frequencies, orient='index').fillna(0).astype(int)
            # Rename columns to identifiers like d1, d2, etc.
            doc_ids = {doc_name: f"d{i+1}" for i, doc_name in enumerate(doc_names)}
            df.rename(columns=doc_ids, inplace=True)
            matrix_data[matrix_type] = df

        for name, df in matrix_data.items():
            if name == "Word Stems":
                st.session_state['df_freq'] = df
            st.write(f"{name} Frequency Matrix:")
            st.dataframe(df)
    else:
        st.error("Please upload documents first using the 'Upload Documents' section.")

elif option == "Indexing Terms":
    st.header("Indexing Terms")
    st.subheader("Decompose the frequency matrix using SVD")

    num_terms_to_retain = st.number_input('Enter the number of terms to retain (K):', min_value=1, max_value=None, value=200)

    if st.button('Perform SVD'):
        if 'df_freq' in st.session_state:
            df_freq = st.session_state['df_freq']  # Retrieve the frequency matrix

            terms = df_freq.index  # Assuming df_freq's index contains the actual term names

            # Perform SVD
            U, Sigma, VT = svd(df_freq, full_matrices=False)

            # Determine the number of terms to retain
            k = min(num_terms_to_retain, len(Sigma))

            # Truncate matrices
            U_k = U[:, :k]
            Sigma_k = Sigma[:k]
            VT_k = VT[:k, :]

            # Display the truncated U matrix with terms
            st.write("Truncated U matrix (Term-Concepts):")
            U_k_df = pd.DataFrame(U_k, index=terms[:len(U)], columns=[f'Document {i}' for i in range(k)])
            st.dataframe(U_k_df)

            # Display the singular values (Sigma) alongside the terms
            st.write("Truncated Sigma (Singular Values):")
            Sigma_k_df = pd.DataFrame({'Singular Value': Sigma_k}, index=terms[:k])
            st.dataframe(Sigma_k_df)

            # Display the truncated VT matrix
            st.write("Truncated VT matrix (Concept-Documents):")
            VT_k_df = pd.DataFrame(VT_k, columns=df_freq.columns)
            st.dataframe(VT_k_df)
        else:
            st.error("Please calculate the frequency matrix first.")

elif option == "Document Query":
    st.header("Document Query using SQL")
    st.subheader("Execute queries to find document similarities, dissimilarities and relevancies")
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
