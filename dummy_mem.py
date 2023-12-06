# pip install openai langchain
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


memory_exists = os.path.isfile("saved_memory.txt")
if memory_exists:
  memory_file = open("saved_memory.txt", "rb")
  saved_mem = memory_file.read()
  memory = pickle.loads(saved_mem)
else:
  memory = ConversationBufferMemory(return_messages=True, ai_prefix="Dobby")
  memory_two = ConversationSummaryMemory(llm=OpenAI(temperature=0), ai_prefix="Dobby")

LVL1_THRESH = 50

def save(memory, name):
  pickled_str = pickle.dumps(memory)
  memory_file = open(name + ".txt", "wb")
  memory_file.write(pickled_str)


def store(input, output):
  memory.save_context(input, output)
  manage_mem()

# memory management
def manage_mem():
  if len(memory.load_memory_variables({})['history']) == LVL1_THRESH:
    history = memory.load_memory_variables({})['history']
  # migrate to level2 -> summarize then cluster by similarity
  for i in range(0, len(history), 2): #assumes input and output go one after the other
    memory_two.save_context({'input': history[i].content}, {'outputs': history[i+1].content})

def clear():
  memory.clear()
  memory_two.clear()

memory.load_memory_variables({})['history']




# grouping
def consolidate_mem(clusters):
# sort the summaries into clusters, where each is an entry in level 2
# classification model? or use an llm here
  return

def load_memory(name, conversation):
  memory_exists = os.path.isfile(name + ".txt")
  if memory_exists:
    memory_file = open(name + ".txt", "rb")
    memory = pickle.loads(memory_file.read())
    # conversation.load_memory_variables(memory)
  else:
    memory = ConversationBufferMemory(return_messages=True, ai_prefix="Dobby")


"""
TODO:
- Clean up interface: personal_interface should initialize it's user-facing GPT, memory_module, and ConversationChain
while memory_module takes care of memory saving at the basic level
- updating with time stamps
- templating with text


"""
