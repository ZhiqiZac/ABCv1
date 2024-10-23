
#import from inbuilt libraries (tiktoken), and personal files (llm.py)
import os
from helper_functions.llm import get_embedding, get_completion, count_tokens 
import tiktoken

'''
There are various ways to import a function 
1. from helper_function import specific function 
2. from helper_function import * 
When using this, need use llm.get embedding, instead of just embedding 
Use the function to explain.  
'''

#import for the link to the password access 
import os
import openai


if load_dotenv('.env'):
   # for local development
   OPENAI_KEY = os.getenv('OPENAI_API_KEY')
else:
   OPENAI_KEY = st.secrets['OPENAI_API_KEY']

# Pass the API Key to the OpenAI Client
client = OpenAI(api_key=OPENAI_KEY)


#all the modules to import 
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


#advanced function 1: mapping a user_query to a process 
def rag_function (user_query): 
   
    #Step A: Find the relevant data from online sources and use 
    
    loader = PyPDFLoader("./data/HDB _ FAQ.pdf") 
    pages = loader.load()

    #Step B: Splitting and chunking 

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,
        chunk_overlap=50,
        length_function=count_tokens
        )
    splitted_documents = text_splitter.split_documents(pages)

    #Step C: Storage: Embedding and vectorstore 

    # An embeddings model is initialized using the OpenAIEmbeddings class.
    # The specified model is 'text-embedding-3-small'.
    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

    # Vectorstore
    vector_store = Chroma.from_documents(
        collection_name="new_files",
        documents=splitted_documents,
        embedding=embeddings_model,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
    )

    #Step D: Retrieval 
    vector_store.similarity_search('Zero Shot', k=3)

    #Step E: Questions and Answer (Function 1)
    qa_chain = RetrievalQA.from_chain_type(
        ChatOpenAI(model='gpt-4o-mini'),
        retriever=vector_store.as_retriever(k=20)
    )
    
    reply = qa_chain.invoke(user_query)
     
    return reply 

#develop a move all rounded function to be all inclusive function 1, function 2, make the reply more understandable based on knowledge of the user" 
#option to expand the function in the future, to improve the accuracy of the results. 
def process_user_message (user_input): 

    return rag_function(user_input)


#more advanced function that makes responses easier to understand + add a liner 
def process_user_message2 (user_input): 

    reply = rag_function(user_input)
    
    prompt = f"""
    Summarise the information into an accurate, yet easy to understand manner. Add a liner "For more information, please visit HDB website for more tips" after an answer
    {reply}
    """
    response = get_completion(prompt)
    return response

'''
#Default LLM: More basic version based on naive LLM model 
#design prompts so that it can generate some sort of response based on the raw input
def summarize (text):
    prompt = f"""
    extract the information that is enclosed in triple backticks into {num_of_sentences}.

    {text}

    
    response = get_completion(prompt)
    return response

'''

'''
Step A: load different kinds of documents which can include : HTML, PDF and csv 
#Type 1: PDF 
#can be either a static data or a link to a web pdf. which is dynamic. 
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("./data/HDB _ FAQ.pdf") 
pages = loader.load()

#functions for future, when they are required. 
#Type 2: HTML 
import bs4
from langchain_community.document_loaders import WebBaseLoader
page_url=["<insert the .html link>"] #accepts multiple inputs for 
pages=loader.load()

#type 3: CSV Files 
from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path="./example_data/mlb_teams_2012.csv")
data = loader.load()


#Retrival Augmented Generation helps to generate higher quality answers compared to default LLM 

#Step A: Find the relevant data from online sources and use 
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("./data/HDB _ FAQ.pdf") 
pages = loader.load()

#Step B: Splitting and chunking 

from langchain.text_splitter import CharacterTextSplitter #not sure if this 

# \n is a newline character
# "\" at the end of the line is used to make sure the text continue on the next line
# It allows long statements to be split across multiple lines for improved readability,
# preventing the need for horizontal scrolling in code editors.


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=500,
    chunk_overlap=50,
    length_function=count_tokens
    )
splitted_documents = text_splitter.split_documents(pages)

#Step C: Storage: Embedding and vectorstore 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# An embeddings model is initialized using the OpenAIEmbeddings class.
# The specified model is 'text-embedding-3-small'.
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')


# For more info on using the Chroma class, refer to the documentation https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/
vector_store = Chroma.from_documents(
    collection_name="new_files",
    documents=splitted_documents,
    embedding=embeddings_model,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
)


#Step D: Retrieval 
vector_store.similarity_search('Zero Shot', k=3)

#Step E: Questions and Answer (Function 1)

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    ChatOpenAI(model='gpt-4o-mini'),
    retriever=vector_store.as_retriever(k=20)
)

#qa_chain.invoke("Why LLM hallucinate?") #example of invoking 
'''
