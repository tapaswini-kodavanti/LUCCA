from gpt_interface import get_completion_from_messages

# pip install openai langchain
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.llms import OpenAI


# first initialize the large language model
llm = OpenAI(
	temperature=0,
	openai_api_key="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",
	model_name="text-davinci-003"
)

# now initialize the conversation chain
conversation = ConversationChain(llm=llm)

memory = ConversationBufferMemory(return_messages=True)
# memory.save_context({"input": "hi"}, {"output": "whats up"})


messages =  [
{'role':'system', 'content':'You are a sarcastic yet funny assistant named dobby'},
{'role':'user', 'content': 'hi my name is emerald and I would like to go to the Anna Hiss Gym'},
{'role':'assistant', 'content': 'uggh you\'re wasting my time. AHG is right down the street'},
{'role':'user', 'content': 'What can i find there?'}
]

response = get_completion_from_messages(messages, temperature=1)
print(response)
#this can sit in a loop
# store info from prompt -> pull from memory -> generate action -> store in mem
# Basic memory

def get_message(message, role):
  new_message = {}
  new_message['role'] = role
  new_message['content'] = message
  return new_message

def get_system_message(message):
  return get_message(message, 'assistant')

def get_user_message(message):
  return get_message(message, 'user')

response_msg = get_system_message(response)
messages.append(response_msg)
memory.save_context({'input': 'What can i find there?'}, {'outputs': response_msg['content']})

messages.append(get_user_message("I see. What are the directions to getting there?"))
print(messages)
response = get_completion_from_messages(messages, temperature=1)
response_msg = get_system_message(response)

memory.save_context({'input': "I see. What are the directions to getting there?"}, {'outputs': response_msg['content']})

response

# memory management
memory_two = ConversationSummaryMemory(llm=OpenAI(temperature=0))

memory.load_memory_variables({})['history']
# memory.load_memory_variables({})['history'][0].content

lvl1_thresh = 50
if len(memory.load_memory_variables({})['history']) == lvl1_thresh:
  history = memory.load_memory_variables({})['history']
  # migrate to level2 -> summarize then cluster by similarity
  for i in range(0, len(history), 2): #assumes input and output go one after the other
    memory_two.save_context({'input': history[i].content}, {'outputs': history[i+1].content})

# grouping
def consolidate_mem(clusters):
  # sort the summaries into clusters, where each is an entry in level 2
  # classification model?
  return