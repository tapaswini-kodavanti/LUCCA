import os
os.environ["OPENAI_API_KEY"]="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"

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.router import MultiRetrievalQAChain
from fact_retrieval import FactRetrieval
from custom_chain import CustomChain
from memory_module import *

# first initialize the large language model
llm = ChatOpenAI(
	temperature=0,
	openai_api_key="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",
	model_name="gpt-3.5-turbo-0613"
)

# take an input
# plug input into chain with memories while adding input into memory

from langchain.prompts.prompt import PromptTemplate

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    Current conversation:
    {history}
    Human: {input}
    Dobby:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
retrieval = FactRetrieval()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# qa = ConversationalRetrievalChain.from_llm(llm, retrieval.vectorstore.as_retriever(search_kwargs={'k': 3}), condense_question_prompt=CONDENSE_QUESTION_PROMPT, memory=memory)
# issue might have to load chroma every time - find a way around this
chat_history = []

retriever_infos = [
    {
        "name": "anna hiss gymnasium",
        "description": "good for answering ut austin lab related questions",
        "retriever": retrieval.vectorstore.as_retriever()
    }
]

prompt_template = PromptTemplate(
    input_variables=['query'],
    template='''
        You are a helpful assistant named Dobby. Please introduce yourself at the beginning of the conversation, and also ask for the name of the
        person that you are interacting with. After this, they will give you a question or comment: {query}
        Respond to this accordingly. 
    '''
)
conversation = MemoryModule()

# chain = MultiRetrievalQAChain.from_retrievers(
#         llm,
#         retriever_infos, 
#         default_chain=conversation.get_conv_chain(),
#         verbose=True
# ) # override the prompt and pass memory dict in -> template doesn't work for everything

def main():
    name_found = False
    name = ""
    while (True):
        # put this in loop
        query = str(input())

        if query == 'quit':
            print('Goodbye!')
            # Save memory...
            conversation.save(conversation.memory, name)
            break
        else:
            # Load in the memory file associated with the person if this is the conversation start
            if not name_found:
                # Check if this is the first time that the person is talking to the robot

                name = query
                name_found = True
                # load_memory(name, conversation)
            # output = conversation.run(query)

            # get_files(query) # runs to update vector store - slow
            # output = qa({"question": query})
            retriever_infos.append({
                "name": "personal memories",
                "description": "good for answering questions about people dobby has interacted with",
                "retriever": conversation.get_personal_retriever(name)
            })
            prompt_template.format(query=query)
            output = conversation.chain.run(query)
            # display
            print(output)

if __name__ == "__main__":
    main()



"""


"""
