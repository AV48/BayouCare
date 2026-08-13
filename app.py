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
    
    messages = [{"role": "system", "content": "You are a friendly chatbot who says a positive comment before replying"}]

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



# --- CSS code for details in Interface ---

my_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="green"
)

custom_css = """
/* MAIN BACKGROUND */
:root, html, body, #root, [class*="gradio-container"] { 
    background-image: linear-gradient(135deg, #70947a 0%, #8095a8 100%) !important;
    background-color: #ffffff !important;
}
div[class*="row"], div[class*="column"], [data-testid="block-container"], .tabs, 
div[class*="gap"], .form, .block, [class*="gr-box"], [class*="gr-panel"], .metadata, 
div[class*="wrapper"], .padded, .gap, .container, .layout, fieldset, [class*="prose"] {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: 90 30 3 3 black; !important;
}
.gradio-container .markdown-text, .gradio-container div[class*="prose"] {
    background-color: transparent !important;
    background: transparent !important;
}
.gradio-container p, .gradio-container h1, .gradio-container h2, .gradio-container span, .gradio-container .markdown-text, .gradio-container label, .gradio-container h3 {
    color: white !important;
}
.chatbot, .message-wrap, .bubble-wrap, div.message-list, div[id="Bachatbot"], .gradio-chatbot, .chat-view {
    background-color: #ffffff !important;
    background: #c5fac0 !important;  
    border: 3px solid #ffffff !important;
    border-radius: 12px !important;
}
.user, [class*="user"], .message.user { 
    background-color: #add6de !important; 
    color: #36c75c !important; 
}
.user p, .user span, .user strong { color: #000000 !important; }
.bot, [class*="bot"], .message.bot, blockquote, pre, code, .prose, 
.bot p, .bot span, .bot strong, .bot li, .bot div { 
    background-color: #cfc8cd !important; 
    background: #cfc8cd !important;
    color: #080708 !important; 
}
.chat-suggestions button, [class*="suggestion"], .chatbot .slots button, .form button.primary, .examples button, .example-btn, button[class*="slot"] {
    background-color: #d4ced6 !important;
    background: #d4ced6 !important;
    color: #000000 !important;
    border: none !important;
    box-shadow: none !important;
}
textarea, div[class*="input-box"], .input-container {
    background-color: #E0CFDB !important;
    background: #E0CFDB !important;
    color: #2A2235 !important;
    border: 1px solid #736686 !important;
    border-radius: 8px !important;
}
textarea::placeholder { color: #736686 !important; opacity: 0.6; }
.submit-button, button[class*="submit"], div[class*="pending"], .generating, [class*="loading"] {
    background-color: #736686 !important;
    background: #736686 !important;
    color: white !important;
}
.message.bot a {
    color: #f5dd07 !important;
    text-decoration: underline !important;
}
div[data-testid="block-container"] img { 
    background: transparent !important; 
    border: none !important; 
    box-shadow: none !important; 
}
"""

# Initialize the interface
with gr.Blocks( ) as demo:


    # 1. Cover Banner (Top)
    cover_image = gr.Image(
        value="Screenshot 2026-08-12 233508.png",
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

        with gr.Column(scale=5):
            gr.Markdown("<h1 style='color:#ffffff; margin: 0;'>BayouCare</h1>")
            gr.Markdown("<p style='color: #6f42c1; font-weight: 500;'> Hey! I'm Bayou, a chatbot designed to help you find low-cost and free health resources near you! Please enter your location and budget available so I can find resources near you! </p>")
""""
        gr.ChatInterface(respond,
                examples=[
                    "What STEM scholarships are available for high school seniors?",
                    "Can you suggest hackathons for beginners?",
                    "How do I find career guidance or mentorship in tech?",
                    "What summer research programs or internships are open now?"
                ],
                cache_examples=False)
""""

# 4. Launch the application
demo.launch(theme=my_theme, css=custom_css)

df = pd.read_csv("Untitled spreadsheet - Sheet1.csv")
if "Hospital": 
    print(df[:546])

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



chatbot = gr.ChatInterface(respond)

chatbot.launch()
my_theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="violet"
)



# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!
