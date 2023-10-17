from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
import pickle
import os

class MemoryModule:
    def __init__(self):
        print('creating memory module object')

    def get_memory(self, name):
        return ConversationBufferMemory(return_messages=True, ai_prefix="Dobby")

    def save(self, conversation, name):
        print("saving conversation")