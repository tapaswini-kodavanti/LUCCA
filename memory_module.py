from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.prompts.prompt import PromptTemplate
from datetime import date
import langchain
import pickle
import os

class MemoryModule:
    input_variables = ["history", "input"]
    def __init__(self):
        print('creating memory module object')

    def get_memory(self, name):
        # Task 2: return a last conversation populate memory object
        return ConversationBufferMemory(return_messages=True, ai_prefix="Dobby")

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


        new_template += "Current conversation: {history} \
                    Human: {input} \
                    Dobby: "

        return PromptTemplate(input_variables = self.input_variables, template = new_template)



    def save(self, memory, name):
        print("saving conversation")

        # STEP 1: Append the conversation directly to the general conversation memory
        history = memory.load_memory_variables({})['history']
        today = date.today()

        # Opening file
        print("PRINTING INFORMATION FOR PERSON " + name + " ON THE DATE OF " + str(today))
        file_name = "general_convo/" + name + ".txt"
        self.write_memory(history, today, file_name, "a")
        # file = open(file_name, "a")
        # file.write(str(today) + "\n")
        
        # for message in history:
        #     m_type = type(message)
        #     if isinstance(message, langchain.schema.messages.HumanMessage):
        #         file.write("person: ")
        #     else:
        #         file.write("ai: ")
        #     file.write(message.content)
        #     file.write("\n")
        # file.write("\n\n")

        # STEP 2: Override the stored memory in the last conversation file
        file_name = "last_convo/" + name + ".txt"
        self.write_memory(history, today, file_name, "w")
        # file = open(file_name, "w")
        # file.write(str(today) + "\n")

        # for message in history:
        #     m_type = type(message)
        #     if isinstance(message, langchain.schema.messages.HumanMessage):
        #         file.write("person: ")
        #     else:
        #         file.write("ai: ")
        #     file.write(message.content)
        #     file.write("\n")
        # file.write("\n\n")

    
    def write_memory(self, history, today, file_name, permissions):
        file = open(file_name, permissions)
        file.write(str(today) + "\n")

        for message in history:
            m_type = type(message)
            if isinstance(message, langchain.schema.messages.HumanMessage):
                file.write("person: ")
            else:
                file.write("ai: ")
            file.write(message.content)
            file.write("\n")
        file.write("\n\n")



        
        # Task 3:
        # Only ask about the last conversation, meaning store the last conversation as natural language
        # directly in another file

        # Task 4: 
        # Figure out how to store declarative memory (one part for last conversation, one part for summarized info)
        # May just end up using a hashmap type of structured database...
        # Use vectorstores and semantic search over the stored information texts

