import os
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from memory_module import MemoryModule
from langchain.prompts.prompt import PromptTemplate

# NOTE: the template, llm, and memory are chained together in a conversation chain when reacting to the humans

# OpenAI API key
os.environ["OPENAI_API_KEY"]="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"

class Interface:
    def __init__(self, llm, template, input_variables):
        self.llm = llm
        self.template = template
        self.prompt = PromptTemplate(input_variables = input_variables, template = self.template)
        self.memory_module = MemoryModule()

        # Will be changed later in the program
        self.name = None
        self.conversation = None
        self.memory = None

    def creat_conv_chain(self, mem):
        self.conversation = ConversationChain(
            prompt = self.prompt,
            llm = self.llm,
            memory = mem, 
            verbose = True
        )

    def save_name(self, name):
        self.name = name

    def save_conv(self):
        self.memory_module.save(self.memory, self.name)

    # TODO: set up the interface so that the first thing the llm asks for is your name. Then, load in the appropriate
    # memory from the memory module

    # note: folder for recent conversations and separate folder for summarized events
    # need to figure out proper representation for indexing into summarized events...
    def run_interface(self):
        # Ask for name
        name = input("Hi! What is your name? ")
        self.save_name(name)

        # Load appropriate memory module
        self.memory = self.memory_module.get_memory(name)
        
        # Activate conversation chain
        self.creat_conv_chain(self.memory)
        print("Hello! I'm Dobby. How can I help you?") # TODO: change how this is formatted
        while (True):
            query = str(input())
            if query == 'quit':
                print('Goodbye!')
                self.save_conv() # Save memory
                break
            else:
                output = self.conversation.run(query)
                print(output)

# take an input
# plug input into chain with memories while adding input into memory

def main():
    # Initialize the large language model
    llm = OpenAI(
        temperature=0,
        openai_api_key="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",
        model_name="text-davinci-003"
    )

    # Initialize the starting template
    template = """The following is a friendly conversation between a human and an AI. The AI is talkative and 
    \\ provides lots of specific details from its context. If the AI does not know the answer to a question, 
    \\ it truthfully says it does not know.\n
        Current conversation: {history}
        Human: {input}
        Dobby:
    """
    input_variables = ["history", "input"]

    # Start up the interface
    interface = Interface(llm, template, input_variables)
    interface.run_interface()


if __name__ == "__main__":
    main()
