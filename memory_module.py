# pip install openai langchain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory

memory = ConversationBufferMemory(return_messages=True, ai_prefix="Dobby")
memory_two = ConversationSummaryMemory(llm=OpenAI(temperature=0), ai_prefix="Dobby")

LVL1_THRESH = 50

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


memory.load_memory_variables({})['history']
# memory.load_memory_variables({})['history'][0].content



# grouping
def consolidate_mem(clusters):
  # sort the summaries into clusters, where each is an entry in level 2
  # classification model? or use an llm here
  return