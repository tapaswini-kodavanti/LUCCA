# pip install chromadb tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from gpt_interface import get_completion
import os
import chromadb

class FactRetrieval:
    embeddings = OpenAIEmbeddings()
    client = chromadb.PersistentClient('chroma_db')
    collection = client.get_or_create_collection("dobby_v1")
    vectorstore = Chroma(
        client=client,
        # collection_name="collection_name",
        embedding_function=embeddings,
    )
    
    def __init__(self) -> None:

        print("initializing fact retrieval")

    def get_file_names(self):
        return os.listdir("memory/factual_db")

    def get_file_embeddings(self, filename):

        if filename != "n/a" and self.vectorstore.get(filename)['ids'] == []:

            loader = TextLoader("memory/factual_db/{filename}".format(filename=filename))
            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.split_documents(documents)
            self.vectorstore.add_documents(documents)
            # vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

    def get_files(self, query):
        PROMPT="""give me just the document name that will best answer the question: {question}
                    options: {options}
                    (if there is no answer, respond with "n/a")
                """.format(question=query, options=self.get_file_names())

        output = get_completion(PROMPT)
        print(output)
        self.get_file_embeddings(output)
    

# get all file names and ask LLM to choose n best files
# stick those n files into retrieval chain and completion

