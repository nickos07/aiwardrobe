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

# 1. Setup & Config
load_dotenv()
client = genai.Client()

# Initialize ChromaDB (This creates a local folder called 'wardrobe_db' to save your items permanently)
chroma_client = chromadb.PersistentClient(path="./wardrobe_db")
# Think of a collection as a table in a database
collection = chroma_client.get_or_create_collection(name="my_wardrobe")

def get_stylist_recommendations(base_item_summary: str):
    prompt = f"""
    The user is building an outfit around this base item: '{base_item_summary}'.
    Act as a high-end personal fashion stylist. 
    Identify 2 complementary clothing categories needed to complete a cohesive outfit.
    For each category, provide a detailed search description that can be used to query a vector database.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StylistPlan,
        )
    )
    return response.text

class ClothingItem(BaseModel):
    # We replace 'str' with Literal to force a strict choice
    category: Literal['top', 'bottom', 'shoes'] = Field(
        description="Strictly categorize the item as 'top' (shirts, sweaters, jackets), 'bottom' (pants, shorts), or 'shoes'."
    )
    primary_color: str = Field(description="The dominant color.")
    pattern: str = Field(description="e.g., solid, striped, floral, graphic print.")
    style: str = Field(description="e.g., casual, formal, streetwear, athletic.")
    season: str = Field(description="The best season: summer, winter, fall, spring, or all-season.")

class OutfitQuery(BaseModel):
    # Force the stylist to ONLY request one of our 3 macro-categories
    needed_category: Literal['top', 'bottom', 'shoes'] = Field(
        description="The category needed, strictly chosen from 'top', 'bottom', or 'shoes'."
    )
    search_prompt: str = Field(description="A descriptive query for vector search, e.g., 'casual light blue relaxed fit denim pants'.")
    styling_reasoning: str = Field(description="Brief explanation of why this complements the base item.")

class StylistPlan(BaseModel):
    recommendations: list[OutfitQuery]

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
                model='gemini-2.5-flash',
                contents=["Analyze this clothing item and extract the requested details.", img],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ClothingItem,
                )
            )
            
            # Parse the JSON string back into a Python dictionary
            metadata = json.loads(response.text)
            
            # Create the Embeddable String (This is what ChromaDB will search against!)
            summary_string = f"A {metadata['style']} {metadata['pattern']} {metadata['primary_color']} {metadata['category']} suitable for {metadata['season']}"
            st.write(f"**Extracted AI Summary:** {summary_string}")
            
            # 4. Insert into the Vector DB
            # We store the 'save_path' in the metadata so our Stylist knows where the image file lives!
            collection.add(
                documents=[summary_string],
                metadatas=[{"image_path": save_path, **metadata}], 
                ids=[file.name] # We use the filename as a unique ID
            )
            
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
            # Display the image using the saved path
            st.image(metadata['image_path'], use_container_width=True)
            st.caption(f"{metadata['category'].title()} - {metadata['primary_color'].title()}")
            
            # 4. The Magic "Style Me" Button
            # We use a unique key for each button so Streamlit knows which one was clicked
            if st.button(f"Style Me! 🪄", key=f"btn_{item_id}"):
                
                st.session_state.selected_item = document
                st.session_state.selected_image = metadata['image_path']

# 5. Display the final Outfit Recommendation
if 'selected_item' in st.session_state:
    st.divider()
    st.header("🎯 Your Curated Outfit")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Base Item**")
        st.image(st.session_state.selected_image, width=200)
        
    with col2:
        with st.spinner("Your AI Stylist is putting together a look..."):
            # Call your function from Milestone 4!
            plan_json = get_stylist_recommendations(st.session_state.selected_item)
            plan_dict = json.loads(plan_json)
            
            st.write("### AI Recommendations:")
            for rec in plan_dict['recommendations']:
                st.write(f"**Need:** {rec['needed_category'].title()}")
                st.write(f"*{rec['styling_reasoning']}*")
                
                # RUN YOUR VECTOR SEARCH HERE (Copy your logic from Milestone 4)
                results = collection.query(
                    query_texts=[rec['search_prompt']],
                    n_results=1,
                    where={"category": rec['needed_category']}
                )
                
                # Display the recommended image!
                if results['distances'][0]:
                    rec_image_path = results['metadatas'][0][0]['image_path']
                    st.image(rec_image_path, width=150)
                else:
                    st.warning(f"No {rec['needed_category']} found in your wardrobe.")
