import os
os.environ["OPENAI_API_KEY"]="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"

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from memory_module import *

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
    {history}
    Human: {input}
    Dobby:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=combined_memory,
)

def main():
    print(memory_exists)
    while (True):
        # put this in loop
        query = str(input())
        if query == 'quit':
            print('Goodbye!')
            break
        else:
            output = conversation.run(query)
            # display
            print(output)

if __name__ == "__main__":
    main()