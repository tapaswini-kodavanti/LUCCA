import os
from langchain.llms import OpenAI
from memory_module import MemoryModule
from langchain.chat_models import ChatOpenAI
from fact_retrieval import FactRetrieval
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool

from langchain import hub

# OpenAI API key
os.environ["OPENAI_API_KEY"]="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"

class Interface:
    def __init__(self, llm):
        self.llm = llm
        self.memory_module = MemoryModule()
        self.retrieval = FactRetrieval()

        self.name = None
        self.conversation = None
        self.chain = None

        self.prompt = '''
            last conversation: {last_conversation} \
            personal information: retrieve from {name}'s memories \
            question: {query} 
        '''
        self.agent = None

    def save_name(self, name):
        self.name = name

    def create_agent(self):
        self.memory_module.init_memory(self.name, self.llm)
        self.tools = [
            create_retriever_tool(
                self.retrieval.vectorstore.as_retriever(), 
                "anna_hiss_gymnasium_or_ahg",
                "Searches and returns documents regarding the ahg or anna hiss gym"
        )]

        prompt = hub.pull("hwchase17/react-chat") # might not use this

        prompt = prompt.partial(
            tools=render_text_description(self.tools),
            tool_names=", ".join([t.name for t in self.tools]),
        )
        llm_with_stop = self.llm.bind(stop=["\nObservation"])

        self.agent = {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x['intermediate_steps']),
            "chat_history": lambda x: x["chat_history"]
        } | prompt | llm_with_stop | ReActSingleInputOutputParser()#| chat_model_with_stop |

    def creat_conv_chain(self):
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True, 
            memory=self.memory_module.memory, 
            handle_parsing_errors=True,
            max_execution_time=1,
            early_stopping_method="generate")


    def save_conv(self):
        self.memory_module.save(self.name)

    def run_interface(self):
        # Ask for name
        name = input("Hi! What is your name? ")
        self.save_name(name)
        
        # Activate conversation chain
        self.create_agent()
        self.creat_conv_chain()
        # print(self.conversation.run("Hi, this is " + name + "!"))
        print(self.agent_executor.invoke({"input": "Hi, this is " + name + "!"})['output'])
        query = str(input())
        while (query != 'quit'):
            # make method to condense to token limit
            output = self.agent_executor.invoke({"input": query})['output']
            # self.conversation.run(query)
            print(output)
            query = str(input())

        print("Goodbye!")
        self.save_conv()


# take an input
# plug input into chain with memories while adding input into memory

def main():
    # Initialize the large language model
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key="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",
        model_name="gpt-3.5-turbo-0613"
    )

    # Start up the interface
    interface = Interface(llm)
    interface.run_interface()


if __name__ == "__main__":
    main()
