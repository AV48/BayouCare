import gradio as gr
from huggingface_hub import InferenceClient
import pandas as pd
#pip install googlemaps
#import os
#import googlemaps


client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", bill_to="kode-with-klossy")
'''
#code to get data from knowledge base & respond -----------------
from sentence_transformers import SentenceTransformer
import torch
with open('Untitled spreadsheet - Sheet1.csv', mode='r', encoding='utf-8') as file:
    knowledge_base = file.read()

text = knowledge_base.strip("Site Name,Services Delivered at Site,Health Center Type,Health Center Location Type,Health Center Location Setting,State,Address,City,ZIP")
chunks = text.split("\n")

# Load the pre-trained embedding model that converts text to vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(chunks):
  # Convert each text chunk into a vector embedding and store as a tensor
  chunk_embeddings = model.encode(chunks, convert_to_tensor=True) # Replace ... with the cleaned_chunks list
  return chunk_embeddings
    
# Call the create_embeddings function and store the result in a new chunk_embeddings variable
chunk_embeddings = create_embeddings(chunks) # Complete this line

def get_top_chunks(query, chunk_embeddings, chunks):
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding_normalized = query_embedding / query_embedding.norm()
    chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)
    similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized)
    top_indices = torch.topk(similarities, k=3).indices

    top_chunks = []
    for i in top_indices: #split top_chunks line into only the name of clinic and the address
        chunk = chunks[i]
        fields = chunk.split(",")
        name = fields[0]
        address = fields[6]
        top_chunks.append((name, address))
    return top_chunks
'''
# chatbot response function
def respond(message, history):
  #  top_results = get_top_chunks(message, chunk_embeddings, chunks)
  #  clinic_info = "\n".join([f"{name} is located at {address}." for name, address in top_results])
    
    messages = [{"role": "system", "content": "You are Bayou, a friendly chatbot that helps patients fine lower cost healthcare in Louisiana. Use the knowledge base provided to answer the question. Initiate conversation by asking if the user needs help first."}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=100,
        temperature = 1
    )
        
    return response.choices[0].message.content.strip()
    
# --- CSS code for details in Interface ---

my_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="green"
)

custom_css = """
/* MAIN BACKGROUND */
:root, html, body, #root, [class*="gradio-container"] { 
    background-image: linear-gradient(90deg,#add8ff 20%, #ffffff 80%) !important;
    background-color: #ffffff !important;
}
div[class*="row"], div[class*="column"], [data-testid="block-container"], .tabs, 
div[class*="gap"], .form, .block, [class*="gr-box"], [class*="gr-panel"], .metadata, 
div[class*="wrapper"], .padded, .gap, .container, .layout, fieldset, [class*="prose"] {
    background-color: transparent !important;
    background: none !important;
    border: none !important;
    box-shadow: 20px 10px 40 green !important;
}
.gradio-container .markdown-text, .gradio-container div[class*="prose"] {
    background-color: transparent !important;
    background: transparent !important;
}
.gradio-container p, .gradio-container h1, .gradio-container h2, .gradio-container span, .gradio-container .markdown-text, .gradio-container label, .gradio-container h3 {
    color: #0c3c69 !important;
}
.chatbot, .message-wrap, .bubble-wrap, div.message-list, div[id="Bachatbot"], .gradio-chatbot, .chat-view, {
    background-color: #000000 !important;
    background: #ffffff !important;  
    border: 3px solid #5cc9ff !important;
    border-radius: 12px !important;
}
.user, [class*="user"], .message.user { 
    background-color: #d0f5d3 !important; 
    color: #000000 !important; 
}
.user p, .user span, .user strong { color: #000000 !important; }
.bot, [class*="bot"], .message.bot, blockquote, pre, code, .prose, 
.bot p, .bot span, .bot strong, .bot li, .bot div { 
    background-color: #000000 !important; 
    background: #b8f2c3 !important;                              
    color: #080708 !important; 
}
.chat-suggestions button, [class*="suggestion"], .chatbot .slots button, .form button.primary, .examples button, .example-btn, button[class*="slot"] {
    background-color: #c6f4f5 !important;
    background: #c6f4f5 !important;
    color: #000000 !important;
    border: none !important;
    box-shadow: 20px 10px 40 #abffa8 !important;
}
textarea, div[class*="input-box"], .input-container {
    background-color: #aef2fc !important;
    background: #aef2fc !important;
    color: #000000 !important;
    border: 1px solid #0a8a7d !important;
    border-radius: 8px !important;
}
textarea::placeholder { color: #a2a4db !important; opacity: 0.6; }
.submit-button, button[class*="submit"], div[class*="pending"], .generating, [class*="loading"] {
    background-color: #7de382 !important;
    background: #ffffff !important;
    color: black !important;
}
.message.bot a {
    color: #f5dd07 !important;
    text-decoration: underline !important;
}
div[data-testid="block-container"] img { 
    background: transparent !important; 
    border: none !important; 
    box-shadow: 20px 30px 40 green !important; 
}
"""

# Initialize the interface
with gr.Blocks( ) as demo:

    # 1. Cover Banner (Top)
    cover_image = gr.Image(
        value="correct one.png",
        show_label="BayouCare",
        container=False,
        height = 200,
        interactive=False
    )

    # 2. Logo & Header Title
    with gr.Row():
        with gr.Column(scale=1, min_width=90):
            logo = gr.Image(
                value="c9062b276c1567f4029e7939837b17ce-louisiana-retro-stroke-usa-states.webp",
                show_label=False,
                width= 200,
                height= 200,
                container=False,
                interactive=False
            )

        with gr.Column(scale=3):
            gr.Markdown("<h1 style='color:#000000; background=#ff00bb;margin: 0;'>BayouCare</h1>")
            gr.Markdown("<p style='color: #000000;background=#ff00bb; font-weight: 500;'> Hey! I'm Bayou, a chatbot designed to help you find low-cost and free health resources near you! Please enter your location and budget available so I can find resources near you! </p>")

        gr.ChatInterface(respond,
                examples=[
                    "Where could I find free healthcare services near 8585 Archives Ave, Baton Rouge, LA 70809?",
                ],
                cache_examples=False)


# 4. Launch the application
demo.launch(theme=my_theme, css=custom_css)

df = pd.read_csv("Untitled spreadsheet - Sheet1.csv")
if "Hospital": 
    print(df[:546])



chatbot = gr.ChatInterface(respond)

chatbot.launch()
''''
my_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="green"
)
'''
