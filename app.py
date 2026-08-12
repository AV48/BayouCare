import gradio as gr
from huggingface_hub import InferenceClient
import pandas as pd
#pip install googlemaps
#import os
#import googlemaps


# This is the same pattern from the Generative AI lesson! It uses the
# Inference Provider API to send your messages to an AI model and get
# a response back. Swap out the model below for a different one if
# you want to experiment!
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", bill_to="kode-with-klossy")
# Note: if this Space doesn't already have one, you'll need to add an
# HF_TOKEN secret in the Space's Settings tab for this to work
# (Settings -> Variables and secrets -> New secret).


def respond(message, history):
    
    messages = [{"role": "system", "content": "You are a friendly chatbot."}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=100,
        temperature = 1
    )
        
    return response.choices[0].message.content.strip()

    print(message)

df = pd.read_csv()


'''
with open("knowledgebase.txt", "r", encoding="utf-8") as file:
    knowledgebase = file.read()

def preprocess_text(text):
    # Strip extra whitespace from the beginning and the end of the text
    cleaned_text = text.strip()
    
    final_cleaned_text = cleaned_text.strip()

    # Split the cleaned_text by every newline character (\n)
    chunks = cleaned_text.split("\n")
    
    # Create an empty list to store cleaned chunks
    cleaned_chunks = []
    
    # Write your for-in loop below to clean each chunk and add it to the cleaned_chunks list
    for chunk in chunks:
        stripped_chunk = chunk.strip()
        if len(stripped_chunk) > 0:
            cleaned_chunks.append(stripped_chunk)

# Print cleaned_chunks
  print(cleaned_chunks)

# Print the length of cleaned_chunks
  print(len(cleaned_chunks))

# Return the cleaned_chunks
  return cleaned_chunks


'''

""""
my_theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="violet"
)

with gr.Blocks( ) as demo:


    # 1. Cover Banner (Top)
    cover_image = gr.Image(
        value="updatedbanner.jpeg",
        show_label=False,
        container=False,
        height=180,
        interactive=False
    )

    # 2. Logo & Header Title
    with gr.Row():
        with gr.Column(scale=1, min_width=80):
            logo = gr.Image(
                value="logo.png",
                show_label=False,
                container=False,
                height=80,
                interactive=False
            )

demo.launch(theme=my_theme, css=custom_css)
"""

chatbot = gr.ChatInterface(respond)

chatbot.launch()


# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!
