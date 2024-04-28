import streamlit as st
import pdfplumber
from collections import Counter, defaultdict
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from numpy.linalg import svd as svd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
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
    
def manhattan_function(doc1_title, doc2_title, VT):
    # Fetch documents from the database
    doc1 = session.query(Document).filter(Document.title == doc1_title).first()
    doc2 = session.query(Document).filter(Document.title == doc2_title).first()

    if not doc1 or not doc2:
        return "One or both documents could not be found."

    # Get document IDs
    id1 = doc1.id - 1  # Adjusting for 0-based index
    id2 = doc2.id - 1  # Adjusting for 0-based index

    # Extract columns corresponding to the document IDs from the transposed matrix VT
    vector1 = VT[:, id1]
    vector2 = VT[:, id2]

    distance = sum(abs(vector1 - vector2))

    return distance

def cosine_function(doc1_title, doc2_title, VT):
    # Fetch documents from the database
    doc1 = session.query(Document).filter(Document.title == doc1_title).first()
    doc2 = session.query(Document).filter(Document.title == doc2_title).first()

    if not doc1 or not doc2:
        return "One or both documents could not be found."

    # Get document IDs adjusted for 0-based indexing
    id1 = doc1.id - 1
    id2 = doc2.id - 1

    # Extract columns corresponding to the document IDs from the transposed matrix VT
    vector1 = VT[:, id1].reshape(1, -1)  # Reshape to 2D array for cosine_similarity
    vector2 = VT[:, id2].reshape(1, -1)  # Reshape to 2D array for cosine_similarity
    st.write(vector1)
    st.write(vector2)

    # Calculate cosine similarity
    similarity = cosine_similarity(vector1, vector2)[0][0]


    return similarity

def inproduct_function(doc1_title, doc2_title, VT):
    
    return 

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
            new_document = Document(title=uploaded_file.name.replace('.pdf', ''), content=full_text)
            session.add(new_document)
        session.commit()  # Commit the transaction to the database
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
    documents = session.query(Document).all()  # Fetch all documents from the database

    if documents:
        doc_titles = [doc.title for doc in documents]  # Use document titles for selection
        selected_titles = st.multiselect('Choose documents', doc_titles, default=doc_titles)

        # Filter documents based on selection
        selected_documents = [doc for doc in documents if doc.title in selected_titles]

        if selected_documents:
            matrix_data = {}
            doc_names = [doc.title for doc in selected_documents]  # Use document titles as identifiers

            for matrix_type, processing_function in document_processing_functions.items():
                doc_frequencies = defaultdict(dict)
                for doc in selected_documents:
                    processed_text = processing_function(doc.content)  # Process text of each document
                    word_count = Counter(processed_text)
                    for word, count in word_count.items():
                        doc_frequencies[word][doc.title] = count

                # Create DataFrame from the nested dictionary
                df = pd.DataFrame.from_dict(doc_frequencies, orient='index').fillna(0).astype(int)
                # Rename columns to identifiers like d1, d2, etc.
                doc_ids = {doc_name: f"d{i+1}" for i, doc_name in enumerate(doc_names)}
                df.rename(columns=doc_ids, inplace=True)
                matrix_data[matrix_type] = df

            tabFreqMatrix, tabStopList, tabSuffixRem, tabWordStem = st.tabs([
                "Original Frequency Matrix", 
                "Stop List Removed Frequency Matrix", 
                "Suffix List Removed Frequency Matrix", 
                "Word Stems Frequency Matrix"
            ])

            # Use columns to center the DataFrame within each tab
            for tab, key in zip(
                [tabFreqMatrix, tabStopList, tabSuffixRem, tabWordStem], 
                ["Original", "Stop List Removed", "Suffix List Removed", "Word Stems"]
            ):
                col1, col2, col3 = tab.columns([1, 8, 1])  # Proportional width: 1:8:1
                with col2:
                    tab.dataframe(matrix_data.get(key, pd.DataFrame()))

            # Optionally store one matrix type for later use
            if "Word Stems" in matrix_data:
                st.session_state['df_freq'] = matrix_data["Word Stems"]
        else:
            st.error("No documents selected. Please select documents first.")
    else:
        st.error("No documents found. Please upload documents first.")

if option == "Indexing Terms":
    st.header("Indexing Terms")
    st.subheader("Decompose the frequency matrix using SVD")

    num_terms_to_retain = st.number_input('Enter the number of terms to retain (K):', min_value=1, max_value=None, value=5)

    if st.button('Perform SVD'):
        if 'df_freq' in st.session_state:
            df_freq = st.session_state['df_freq']  # Retrieve the frequency matrix

            terms = df_freq.index  # Assuming df_freq's index contains the actual term names

            # Initialize TruncatedSVD
            svd_model = TruncatedSVD(n_components=num_terms_to_retain, random_state=42)
            U_k = svd_model.fit_transform(df_freq)  # Fit and transform the frequency matrix to get U_k

            # Perform SVD
            Us, Sigmas, VTs = svd(df_freq, full_matrices=False)

            # Determine the number of terms to retain
            k = min(num_terms_to_retain, len(Sigmas))

            # Truncate matrices
            U_k = Us[:, :k]

            # Invert the sign of each element in U_k
            U_k = -U_k

            # Sigma values are accessible through svd_model.singular_values_
            Sigma_k = svd_model.singular_values_

            # VT_k is accessible through svd_model.components_
            VT_k = svd_model.components_

            st.session_state['VT'] = VT_k

            # Display the truncated U matrix with terms
            st.write("Truncated U matrix (Term-Concepts):")
            U_k_df = pd.DataFrame(U_k, index=terms, columns=[f'Concept {i+1}' for i in range(num_terms_to_retain)])
            st.dataframe(U_k_df)

            # Display the singular values (Sigma)
            st.write("Truncated Sigma (Singular Values):")
            Sigma_k_df = pd.DataFrame({'Singular Value': Sigma_k}, index=[f'Concept {i+1}' for i in range(len(Sigma_k))])
            st.dataframe(Sigma_k_df)

            # Display the truncated VT matrix
            st.write("Truncated VT matrix (Concept-Documents):")
            VT_k_df = pd.DataFrame(VT_k, columns=df_freq.columns)
            st.dataframe(VT_k_df)
        else:
            st.error("Please calculate the frequency matrix first.")


elif option == "Document Query":
    st.header("Document Query using SQL")
    st.subheader("Execute queries to find document similarities, dissimilarities, and relevancies")
    
    # Dropdown for choosing similarity function
    function = st.selectbox(
        "Choose a function",
        ("Cosine Similarity", "Manhattan Distance","Internal product Similarity")
    )
    
    VT = st.session_state['VT']

    # Text input for document names/IDs
    doc1 = st.text_input("Enter the first document name/id:", key="doc1")
    doc2 = st.text_input("Enter the second document name/id:", key="doc2")

    if st.button("Evaluation between Documents"):
        if function == "Cosine Similarity":
            similarity = cosine_function(doc1, doc2, VT)
            st.write(f"Similarity result between documents {doc1} and {doc2}: {similarity}") 
        elif function == "Manhattan Distance":
            dissimilarity = manhattan_function(doc1, doc2, VT)
            st.write(f"Dissimilarity result between documents {doc1} and {doc2}: {dissimilarity:.4f}") 
        elif function == "Internal product Similarity":
            similarity = inproduct_function(doc1, doc2, VT)
            st.write(f"Similarity result between documents {doc1} and {doc2}: {similarity:.4f}") 

    st.write("### Document Relevancy")
    query = st.text_input("Enter your query:", key="query")
    n_docs = st.slider("Select number of documents", 1, 10, 5, key="n_docs")
    if st.button("Fetch Documents Based on Query"):
        relevant_documents = fetch_query(query, n_docs)
        st.write(f"Fetching {n_docs} most relevant documents for the query: `{query}`")
        for doc in relevant_documents:
            st.write(doc)
