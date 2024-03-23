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
import re
import os

class MemoryModule:
    STM_LIMIT=50 # TODO: use this
    input_variables = ["chat_history", "input"]
    def __init__(self):
        print('creating memory module object')
        self.prompt = None
        self.memory = None
        self.summarizer = None


    def get_memory(self, name):
        # Task 2: return a last conversation populate memory object
        return ConversationBufferMemory(return_messages=True, ai_prefix="Dobby", memory_key="chat_history")

    def get_prompt(self, name):
        memory_exists = os.path.isfile("last_convo/" + name + ".txt")
        
        new_template = "You are a helpful chatbot, LUCCA, designed to hold casual conversations with people and provide them with \
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
        self.summarizer = ConversationSummaryMemory(llm=llm)
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
            loader = TextLoader(file_name) # TODO: load the summarized version here too?
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

    def summarize_memory(self, file_name, text, permissions):
        # summarize memory -> option 1 summarize entire file, option 2 summarize entire conversaton
        self.parse_conversation(text)
        # store in long-term
        file = open(file_name, permissions)
        for interaction in self.summarizer.chat_memory:
            if isinstance(interaction, langchain.schema.messages.SystemMessage):
                file.write("summary: ")
            file.write(interaction.content)
            file.write("\n")
        file.write("\n\n")

        # clear general_convo
        open(file_name, "w").close()
        
    def parse_conversation(self, text):
        # Split the text into lines
        lines = text.strip().split('\n')
        
        # Initialize variables to store input-output tuples
        current_tuple = {'input': '', 'output': ''}

        for line in lines:
            # Extract date
            if re.match(r'\d{4}-\d{2}-\d{2}', line):
                current_tuple['date'] = line
            else:
                # Split the line into speaker and message
                speaker, message = map(str.strip, line.split(':', 1))
                
                # Determine if it's an input or output
                if speaker.lower() == 'person':
                    current_tuple['input'] += message + ' '
                elif speaker.lower() == 'ai':
                    current_tuple['output'] += message + ' '

        # Append the last tuple
        self.summarizer.save_context({'input': current_tuple['input'].strip()}, {'output': current_tuple['output'].strip()})
