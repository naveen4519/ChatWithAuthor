from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import speech_recognition as sr
import pyttsx3
from translate import Translator
from pymongo import MongoClient
import gridfs

# Load environment variables and configure Google API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize NLTK
nltk.download('stopwords')
stop_words = stopwords.words('english')
custom_stopwords = ["what", "is", "how", "who", "explain", "about", "?", "please", "hey", "whatsup", "can u explain", "where", "why", "tell"]
stop_words.extend(custom_stopwords)

# Initialize MongoDB
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["pdf_chat"]
fs = gridfs.GridFS(db)

app = Flask(__name__)
CORS(app)

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)  # Reduced chunk size
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context and also give detailed information on the question. If the question is out of context, say "It was out of the context but here is the answer for it" before providing the answer. Make sure to provide all the details. If the answer is not in the provided context just say "naveen", don't give any other answer, don't give wrong answer.\n\n
    Context:\n{context}\n
    Question: \n{question}\n

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to calculate cosine similarity
def calculate_cosine_similarity(text, user_question):
    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    tfidf_matrix = vectorizer.fit_transform([text, user_question])
    cos_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return cos_similarity

@app.route('/upload', methods=['POST'])
def upload_pdf():
    files = request.files.getlist("pdf_files")
    for file in files:
        fs.put(file, filename=file.filename)
    return jsonify({"message": "Files uploaded successfully"}), 200

@app.route('/process', methods=['POST'])
def process_pdf():
    pdf_docs = [fs.get_last_version(filename) for filename in request.json['filenames']]
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return jsonify({"message": "PDF processed successfully"}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    user_question = data['question']
    language_code = data['language_code']

    existing_entry = db.history.find_one({"question": user_question, "language_code": language_code})
    
    if existing_entry:
        translated_response = existing_entry["answer"]
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        combined_text = " ".join([doc.page_content for doc in docs])
        
        # Limit combined text length
        if len(combined_text) > 5000:
            combined_text = combined_text[:5000]
        
        score = calculate_cosine_similarity(combined_text, user_question)

        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        translator = Translator(to_lang=language_code)

        if "naveen" in response["output_text"]:
            if score > 0.00125:
                model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                response = model.invoke(user_question)
                translated_response = translator.translate(response.content)
            else:
                model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                response = model.invoke(user_question)
                translated_response = translator.translate(response.content)
        else:
            translated_response = translator.translate(response["output_text"])

        db.history.insert_one({"question": user_question, "answer": translated_response, "language_code": language_code})
    
    return jsonify({"response": translated_response}), 200

@app.route('/history', methods=['GET'])
def get_history():
    history = list(db.history.find({}, {"_id": 0}).sort([('$natural', -1)]))
    return jsonify(history), 200

if __name__ == '__main__':
    app.run(debug=False)
