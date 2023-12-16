import openai
from time import sleep

def get_embedding(text):
    while True:
        try:
            response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
            break
        except Exception as e:
            print("Error encountered:", e)
            sleep(10)
    return response["data"][0]["embedding"]
