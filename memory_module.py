from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.prompts.prompt import PromptTemplate
from datetime import datetime
import langchain
import pickle
import os

class MemoryModule:
    input_variables = ["chat_history", "input"]
    def __init__(self):
        print('creating memory module object')
        self.prompt = None
        self.memory = None

    def get_memory(self, name):
        # Task 2: return a last conversation populate memory object
        return ConversationBufferMemory(return_messages=True, ai_prefix="Dobby", memory_key="chat_history")

    def get_prompt(self, name):
        memory_exists = os.path.isfile("last_convo/" + name + ".txt")
        
        # TODO: change this
        new_template = "You are a helpful chatbot designed to hold casual conversations with people and provide them with \
            information when you can. A member of the lab, " + name + ", has had conversations with you over the past few weeks \
            in which they've told you information about their life. I will provide information about the previous conversation. \
            What is your very concise response? You don't need to mention everything they've told you about. Just respond \
            casually and perhaps ask about an update on something they've told you about. \
            Previous conversation: "
        test_template = "The following is a friendly conversation between a human and an AI. The AI is talkative and \
            provides lots of specific details from its context. If the AI does not know the answer to a question, \
            it truthfully says it does not know.\n \
            Previous conversation: "

        if memory_exists:
            print("getting last stored conversation")
            file_name = "last_convo/" + name + ".txt"
            file = open(file_name, "r")
            info = file.read()
            new_template += info
        else: # this is the first time the person is talking to the robot
            print("we are not adding any new information")
            new_template += ""


        new_template += "Current conversation: {chat_history} \
                    Human: {input} \
                    Dobby: "

        return PromptTemplate(input_variables = self.input_variables, template = new_template)

    def get_conv_chain(self, name, llm):
        self.prompt = self.get_prompt(name)
        self.memory = self.get_memory(name)
        conversation = ConversationChain(
            prompt = self.prompt,
            llm = llm,
            memory = self.memory, 
            verbose = True
        )

        return conversation
    
    def init_memory(self, name, llm):
        self.prompt = self.get_prompt(name)
        self.memory = self.get_memory(name)

    # Returns a retriever object for personal memory
    def get_personal_retriever(self, name):
        file_name = "general_convo/" + name + ".txt"
        memory_exists = os.path.isfile(file_name)
        if memory_exists:
            loader = TextLoader(file_name)
            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(documents, embeddings)
            return vectorstore
        else:
            return None

    def save(self, name):
        print("saving conversation")

        # STEP 1: Append the conversation directly to the general conversation memory
        history = self.memory.load_memory_variables({})['chat_history']
        timestamp = datetime.now().strftime("%d/%m/%Y at %H:%M:%S")

        # Opening file
        print("PRINTING INFORMATION FOR PERSON " + name + " ON THE DATE OF " + str(timestamp))
        file_name = "general_convo/" + name + ".txt"
        self.write_memory(history, timestamp, file_name, "a")

        # STEP 2: Override the stored memory in the last conversation file
        file_name = "last_convo/" + name + ".txt"
        self.write_memory(history, timestamp, file_name, "w")

    
    def write_memory(self, history, timestamp, file_name, permissions):
        file = open(file_name, permissions)
        file.write("* Conversation on " + str(timestamp) + "\n")

        for message in history:
            m_type = type(message)
            if isinstance(message, langchain.schema.messages.HumanMessage):
                file.write("person: ")
            else:
                file.write("ai: ")
            file.write(message.content)
            file.write("\n")
        file.write("\n\n")
