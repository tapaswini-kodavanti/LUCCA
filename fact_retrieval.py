# pip install chromadb tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from gpt_interface import get_completion
import os
import chromadb

embeddings = OpenAIEmbeddings()
client = chromadb.PersistentClient('chroma_db')
collection = client.get_or_create_collection("dobby_v1")
vectorstore = Chroma(
    client=client,
    # collection_name="collection_name",
    embedding_function=embeddings,
)

# get all file names and ask LLM to choose n best files
# stick those n files into retrieval chain and completion

def get_file_names():
    return os.listdir("memory/factual_db")

def get_file_embeddings(filename):

    if filename != "" and vectorstore.get(filename)['ids'] == []:

        loader = TextLoader("memory/factual_db/{filename}".format(filename=filename))
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(documents)
        vectorstore.add_documents(documents)
        # vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

def get_files(query):
    PROMPT="""give me just the document name that will best answer the question: {question}
                options: {options}
            """.format(question=query, options=get_file_names())

    output = get_completion(PROMPT)
    print(output)
    get_file_embeddings(output)
    