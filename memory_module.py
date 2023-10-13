# pip install openai langchain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ChatMessageHistory, CombinedMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
import pickle
import os

memory_exists = os.path.isfile("saved_memory.txt")
if memory_exists:
  print("Opening")
  memory_file = open("saved_memory.txt", "rb")
  saved_mem = memory_file.read()
  memory = pickle.loads(saved_mem)
else:
  memory = ConversationBufferMemory(return_messages=True, ai_prefix="Dobby")
memory_two = ConversationSummaryMemory(llm=OpenAI(temperature=0), ai_prefix="Dobby")
# combined_memory = CombinedMemory(memories=[memory, memory_two]) # make custom class to choose the best and make memory queries

LVL1_THRESH = 50

def save(memory):
  pickled_str = pickle.dumps(memory)
  memory_file = open("saved_memory.txt", "wb")
  memory_file.write(pickled_str)


def store(input, output):
  memory.save_context(input, output)
  manage_mem()

# memory management
def manage_mem():
  # if len(memory.load_memory_variables({})['history']) == LVL1_THRESH:
  #   history = memory.load_memory_variables({})['history']
  # # migrate to level2 -> summarize then cluster by similarity
  # for i in range(0, len(history), 2): #assumes input and output go one after the other
  #   memory_two.save_context({'input': history[i].content}, {'outputs': history[i+1].content})
  print("Should be managing memory now")

def clear():
  memory.clear()
  memory_two.clear()

# memory.load_memory_variables({})['history'][0].content

def show_mem():
  print(memory.load_memory_variables({})['history'])


# grouping
def consolidate_mem(clusters):
  # sort the summaries into clusters, where each is an entry in level 2
  # classification model? or use an llm here
  return