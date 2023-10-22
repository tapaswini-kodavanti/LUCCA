import os
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from memory_module import MemoryModule
from langchain.prompts.prompt import PromptTemplate

# OpenAI API key
os.environ["OPENAI_API_KEY"]="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"

class Interface:
    def __init__(self, llm):
        self.llm = llm
        self.memory_module = MemoryModule()

        self.name = None
        self.conversation = None

    def save_name(self, name):
        self.name = name

    def creat_conv_chain(self):
        self.conversation = self.memory_module.get_conv_chain(self.name, self.llm)

    def save_conv(self):
        self.memory_module.save(self.name)

    def run_interface(self):
        # Ask for name
        name = input("Hi! What is your name? ")
        self.save_name(name)
        
        # Activate conversation chain
        self.creat_conv_chain()
        print(self.conversation.run("Hi, this is " + name + "!"))
        query = str(input())
        while (query != 'quit'):
            output = self.conversation.run(query)
            print(output)
            query = str(input())

        print("Goodbye!")
        self.save_conv()


# take an input
# plug input into chain with memories while adding input into memory

def main():
    # Initialize the large language model
    llm = OpenAI(
        temperature=0,
        openai_api_key="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",
        model_name="gpt-3.5-turbo-0613"
    )

    # Start up the interface
    interface = Interface(llm)
    interface.run_interface()


if __name__ == "__main__":
    main()
