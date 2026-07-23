import streamlit as st
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import chromadb
from PIL import Image
from typing import Literal
import time

# 1. Setup & Config
load_dotenv()
client = genai.Client()

# Initialize ChromaDB (This creates a local folder called 'wardrobe_db' to save your items permanently)
chroma_client = chromadb.PersistentClient(path="./wardrobe_db")
# Think of a collection as a table in a database
collection = chroma_client.get_or_create_collection(name="my_wardrobe")

class ClothingItem(BaseModel):
    # 1. The Macro-Category (For strict ChromaDB filtering)
    category: Literal['top', 'bottom', 'shoes'] = Field(
        description="Strictly categorize the item as 'top', 'bottom', or 'shoes' for database filtering."
    )
    # 2. The Micro-Category (For semantic vector search)
    item_type: str = Field(
        description="The specific garment type (e.g., 't-shirt', 'sweater', 'button-down', 'jeans', 'chinos', 'loafers')."
    )
    primary_color: str = Field(description="The dominant color.")
    pattern: str = Field(description="e.g., solid, striped, floral, graphic print.")
    style: str = Field(description="e.g., casual, formal, streetwear, athletic.")
    season: str = Field(description="The best season: summer, winter, fall, spring, or all-season.")

def get_stylist_recommendations(base_item_summary: str):
    prompt = f"""
    The user is building an outfit around this base item: '{base_item_summary}'.
    Act as a high-end personal fashion stylist. 
    Create EXACTLY 2 completely distinct outfit combinations around this base item (e.g., one casual, one more elevated).
    For each outfit, identify the complementary clothing categories needed to complete it.
    Use simple, primary color names in your search prompts (e.g., use 'white', 'beige', or 'brown' instead of 'camel', 'ecru', or 'taupe').
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StylistPlan, # Now this works because StylistPlan is defined above!
        )
    )
    return response.text

class OutfitQuery(BaseModel):
    needed_category: Literal['top', 'bottom', 'shoes'] = Field(
        description="The category needed, strictly chosen from 'top', 'bottom', or 'shoes'."
    )
    search_prompt: str = Field(
        # Notice we changed [category] to [item_type] in the instructions here!
        description="Strict format required: 'A [style] [pattern] [color] [item_type] suitable for [season]'. Example: 'A casual solid cream t-shirt suitable for summer'."
    )
    styling_reasoning: str = Field(description="Brief explanation of why this complements the base item.")

# --- NEW SCHEMA ---
class OutfitOption(BaseModel):
    outfit_name: str = Field(description="A catchy name for this look (e.g., 'Weekend Casual' or 'Elevated Evening').")
    outfit_vibe: str = Field(description="A short sentence describing the overall aesthetic.")
    items: list[OutfitQuery] 

class StylistPlan(BaseModel):
    outfits: list[OutfitOption]

st.title("AI Digital Wardrobe Setup")

# 2. The Streamlit File Uploader
uploaded_files = st.file_uploader("Upload your clothes to the wardrobe", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if st.button("Process & Add to Wardrobe") and uploaded_files:
    # Create a folder to permanently save the uploaded images
    os.makedirs("wardrobe_images", exist_ok=True)

    for file in uploaded_files:
        # Save the file to our local folder so we can display it later in Milestone 4
        save_path = f"wardrobe_images/{file.name}"
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
            
        img = Image.open(save_path)
        st.image(img, width=150, caption=f"Processing {file.name}...")
        
        # 3. Call Gemini to get the Structured JSON
        with st.spinner("Extracting features with Gemini..."):
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=["Analyze this clothing item and extract the requested details.", img],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ClothingItem,
                )
            )
            
            # Parse the JSON string back into a Python dictionary
            metadata = json.loads(response.text)
            
            # Create the Embeddable String (This is what ChromaDB will search against!)
            # Create the Embeddable String using the new 'item_type'
            summary_string = f"A {metadata['style']} {metadata['pattern']} {metadata['primary_color']} {metadata['item_type']} suitable for {metadata['season']}"
            st.write(f"**Extracted AI Summary:** {summary_string}")
            
            # 4. Insert into the Vector DB
            # We store the 'save_path' in the metadata so our Stylist knows where the image file lives!
            collection.add(
                documents=[summary_string],
                metadatas=[{"image_path": save_path, **metadata}], 
                ids=[file.name] # We use the filename as a unique ID
            )
        st.info("Throttling API to respect rate limits... waiting 4 seconds.")
        time.sleep(4)
            
    st.success("All items successfully added to your Digital Wardrobe! 🚀")

st.divider()
st.header("Your Digital Wardrobe")

# 1. Fetch everything from ChromaDB
all_items = collection.get()

if not all_items['ids']:
    st.info("Your wardrobe is empty! Upload some items above.")
else:
    # 2. Create a grid layout (e.g., 3 columns)
    cols = st.columns(3)
    
    # 3. Loop through your items and display them
    for idx, (item_id, metadata, document) in enumerate(zip(all_items['ids'], all_items['metadatas'], all_items['documents'])):
        # Alternate which column we place the item in
        col = cols[idx % 3] 
        
        with col:
            # Display your wardrobe image
            st.image(metadata['image_path'], use_container_width=True)
            st.caption(f"{metadata['category'].title()} - {metadata['primary_color'].title()}")
            
            # --- THE INFERENCE BLOCK ---
            # This code ONLY runs the exact second you click the button.
            if st.button(f"Style Me! 🪄", key=f"btn_{item_id}"):
                
                # 1. Save the image info to memory
                st.session_state.selected_item = document
                st.session_state.selected_image = metadata['image_path']
                
                # 2. Call the 2.5-flash API exactly ONCE
                with st.spinner("Your AI Stylist is putting together a look..."):
                    plan_json = get_stylist_recommendations(document)
                    
                    # 3. Save the expensive API response to memory!
                    st.session_state.current_plan = json.loads(plan_json)

# --- THE RENDER BLOCK ---
# This sits completely outside the columns/buttons.
# It reads for FREE from your local memory, never calling Google!
# --- THE RENDER BLOCK ---
if 'current_plan' in st.session_state:
    st.divider()
    st.header("🎯 Your Curated Outfits")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Base Item**")
        st.image(st.session_state.selected_image, width=200)
        
    with col2:
        st.write("### AI Recommendations")
        
        # 1. Create a list of tab names from the AI's generated outfits
        tab_names = [outfit['outfit_name'] for outfit in st.session_state.current_plan['outfits']]
        
        # 2. Generate the Streamlit Tabs
        outfit_tabs = st.tabs(tab_names)
        
        # 3. Loop through the tabs and populate them
        for idx, tab in enumerate(outfit_tabs):
            with tab:
                # Grab the specific outfit data for this tab
                outfit_data = st.session_state.current_plan['outfits'][idx]
                st.write(f"*{outfit_data['outfit_vibe']}*")
                st.divider()
                
                # Now loop through the items exactly like you did before!
                for rec in outfit_data['items']:
                    st.write(f"**Need:** {rec['needed_category'].title()}")
                    st.write(f"*{rec['styling_reasoning']}*")
                    
                    # Local ChromaDB Vector Search
                    results = collection.query(
                        query_texts=[rec['search_prompt']],
                        n_results=1,
                        where={"category": rec['needed_category']}
                    )
                    
                    if results['distances'][0]:
                        rec_image_path = results['metadatas'][0][0]['image_path']
                        st.image(rec_image_path, width=150)
                    else:
                        st.warning(f"No {rec['needed_category']} found in your wardrobe.")
