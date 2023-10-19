from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from datetime import date
import langchain
import pickle
import os

class MemoryModule:
    def __init__(self):
        print('creating memory module object')

    def get_memory(self, name):
        return ConversationBufferMemory(return_messages=True, ai_prefix="Dobby")

    def save(self, memory, name):
        print("saving conversation")

        # Task 1:
        # Store the conversation text with time as text directly in the .txt file
        # Only ask about the last conversation
        history = memory.load_memory_variables({})['history']
        today = date.today()

        # Opening file
        file_name = name + ".txt"
        file = open(file_name, "a")
        file.write(str(today) + "\n")

        print("PRINTING INFORMATION FOR PERSON " + name + " ON THE DATE OF " + str(today))
        for message in history:
            m_type = type(message)
            if isinstance(message, langchain.schema.messages.HumanMessage):
                file.write("person: ")
            else:
                file.write("ai: ")
            file.write(message.content)
            file.write("\n")
        file.write("\n\n")

        

        # Task 2: 
        # Figure out how to store declarative memory (one part for last conversation, one part for summarized info)
        # May just end up using a hashmap type of structured database...


        # 
        # Summarizing inputs to 50 messages