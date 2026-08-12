import gradio as gr
from huggingface_hub import InferenceClient
import pandas as pd
import random
#pip install googlemaps
#import os
#import googlemaps


# This is the same pattern from the Generative AI lesson! It uses the
# Inference Provider API to send your messages to an AI model and get
# a response back. Swap out the model below for a different one if
# you want to experiment!
#
# Note: if this Space doesn't already have one, you'll need to add an
# HF_TOKEN secret in the Space's Settings tab for this to work
# (Settings -> Variables and secrets -> New secret).

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", bill_to="kode-with-klossy")

def positive():
    
    comments = ["Asking for help is a sign of self-respect and self-awareness.","Changing my mind is a strength, not a weakness.","I am loved and worthy.",
            "I look forward to tomorrow and the opportunities that await me.", " I will allow myself to evolve.","There is poetry in everything, if I look for it.","When I talk to myself as I would a friend, I see all my best qualities and I allow myself to shine."
            "When I focus on my reason for being, I am infinitely brave.","Today is an opportunity to grow and learn.","Saying “no” is an act of self-affirmation, too.","My heart knows its own way.","Letting go creates space for opportunities to come."]

    print(random.choice(comments)) 

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

""""
with open("knowledgebase.txt", "r", encoding="utf-8") as file:
    knowledgebase = file.read()

def preprocess_text(text):
    # Strip extra whitespace from the beginning and the end of the text
    cleaned_text = text.strip()
    
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

# Call the preprocess_text function and store the result in a cleaned_chunks variable
cleaned_chunks = preprocess_text(knowledgebase) # Complete this line

pd.read_csv(cleaned_chunks)
"""
chatbot = gr.ChatInterface(positive, respond)

chatbot.launch()


# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!
