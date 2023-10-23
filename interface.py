import os
os.environ["OPENAI_API_KEY"]="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"

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from memory_module import memory_exists, memory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# from fact_retrieval import get_files, vectorstore
from langchain.chains.router import MultiRetrievalQAChain
from fact_retrieval import FactRetrieval

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
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=memory,
)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
retrieval = FactRetrieval()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retrieval.vectorstore.as_retriever(search_kwargs={'k': 3}), condense_question_prompt=CONDENSE_QUESTION_PROMPT, memory=memory)
# issue might have to load chroma every time - find a way around this
chat_history = []

retriever_infos = [
    {
        "name": "anna hiss gymnasium",
        "description": "good for answering ut austin lab related questions",
        "retriever": retrieval.vectorstore.as_retriever()
    },
]

prompt_template = PromptTemplate(
    input_variables=['query'],
    template='''
        You are a helpful assistant named Dobby. Please introduce yourself at the beginning of the conversation, and also ask for the name of the
        person that you are interacting with. After this, they will give you a question or comment: {query}
        Respond to this accordingly. 
    '''
)

prompt_template.format(query="")
chain = MultiRetrievalQAChain.from_retrievers(OpenAI(), retriever_infos, verbose=True) # override the prompt and pass memory dict in -> template doesn't work for everything
def main():
    name_found = False
    name = ""
    while (True):
        # put this in loop
        query = str(input())

        if query == 'quit':
            print('Goodbye!')
            # Save memory...
            save(conversation.memory, name)
            break
        else:
            # Load in the memory file associated with the person if this is the conversation start
            if not name_found:
                # Check if this is the first time that the person is talking to the robot

                name = query
                name_found = True
                load_memory(name, conversation)
            # output = conversation.run(query)

            # get_files(query) # runs to update vector store - slow
            # output = qa({"question": query})
            prompt_template.format(query=query)
            output = chain.run(query)
            # display
            print(output)

if __name__ == "__main__":
    main()



"""


"""
