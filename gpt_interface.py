import openai
import json
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)
import pandas as pd

API_KEY="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"

model = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

# reading API_KEY from the environment file
api_key_openai = API_KEY
openai.api_key = api_key_openai

def get_completion(prompt, model=model, max_tokens=150):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_completion_from_messages(messages, model=model, temperature=0, max_tokens=150):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    # print(str(response.choices[0].message))
    return response.choices[0].message["content"]


def get_response(prompt, model=model, max_tokens=150):
    response = openai.ChatCompletion.create(
        engine=model, #or "your_chosen_engine",
        prompt=prompt,
        max_tokens=max_tokens,
        n=5,
        stop=None,
        temperature=0.8,
    )

    return response.choices[0].text.strip()


# create dataframe
# df = pd.DataFrame(columns=['Name', 'Content'])
# short_term = pd.DataFrame(columns=['Data']) # we can make this specific to things about the building

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

