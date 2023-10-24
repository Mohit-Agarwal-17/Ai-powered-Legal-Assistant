import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
import pickle
import speech_recognition as sr

# Function to load PDFs and create vector store
@st.cache_data
def load_or_generate_vectorstore():
    print("Loading OR Generating Chunks")
    try:
        # Attempt to load the stored vector store
        with open('vector_store.pkl', 'rb') as file:
            vector_store = pickle.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, generate and store the vector store
        loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
        text_chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cuda"})

        vector_store = FAISS.from_documents(text_chunks, embeddings)

        # Save the generated vector store
        with open('vector_store.pkl', 'wb') as file:
            pickle.dump(vector_store, file)

    return vector_store

print("Chunks Generated")
# Create vector store
vector_store = load_or_generate_vectorstore()


# Create llm
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama",
                    config={'max_new_tokens': 128, 'temperature': 0.01})
print("Model Loaded")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory)

summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn")

# Initialize the recognizer
recognizer = sr.Recognizer()

st.title("COI ChatBot")

def handle_audio_input():
    with sr.Microphone() as source:
        st.write("Listening...")
        audio_data = recognizer.listen(source)
        st.write("Processing...")

        try:
            # Recognize speech using Google Speech Recognition
            user_input = recognizer.recognize_google(audio_data)
            st.write(f"User Input (Speech): {user_input}")
            return user_input
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand your audio.")
            return None

def conversation_chat(query):
    if "Summarize:" in query:
        result = chain({"question": query[len("Summarize:"):], "chat_history": st.session_state['history']})
        summarized_response = summarizer(result["answer"], max_length=50, min_length=10, do_sample=False)[0][
            'summary_text']
        st.session_state['history'].append((query, summarized_response))
        return summarized_response
    else:
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask your question", key='input')
            print(user_input)
            submit_button = st.form_submit_button(label='Send')
        if submit_button and user_input:
            output = conversation_chat(user_input)
            print(output)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(f"User: {st.session_state['past'][i]}", is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                if "Summarize:" in st.session_state['past'][i]:
                    message("ChatBot (Summarized): " + st.session_state['generated'][i], key=str(i),
                            avatar_style="fun-emoji")
                else:
                    message("ChatBot: " + st.session_state['generated'][i], key=str(i), avatar_style="fun-emoji")

# Initialize session state
initialize_session_state()

# Voice input button
voice_button = st.button("Voice Input")

if voice_button:
    user_input = handle_audio_input()
    print(user_input)
    if user_input:
        output = conversation_chat(user_input)
        print(output)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

# Display chat history
display_chat_history()
