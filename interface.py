import os
os.environ["OPENAI_API_KEY"]="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"

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from memory_module import combined_memory, show_mem
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from fact_retrieval import vectorstore
# first initialize the large language model
llm = OpenAI(
	temperature=0,
	openai_api_key="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",
	model_name="text-davinci-003"
)

# take an input
# plug input into chain with memories while adding input into memory

from langchain.prompts.prompt import PromptTemplate

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    Current conversation:
    {recent_history}
    {history}
    Human: {input}
    Dobby:
"""
PROMPT = PromptTemplate(input_variables=["recent_history", "history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=combined_memory,
)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), condense_question_prompt=CONDENSE_QUESTION_PROMPT, memory=memory)

chat_history = []

def main():
    while (True):
        # put this in loop
        query = str(input())
        if query == 'quit':
            print('Goodbye!')
            show_mem()
            break
        else:
            # output = conversation.run(query)
            output = qa({"question": query})
 
            # display
            print(output['answer'])

if __name__ == "__main__":
    main()