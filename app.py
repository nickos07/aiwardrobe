import streamlit as st
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import chromadb
from PIL import Image

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

# Our Pydantic Schema from Milestone 2
class ClothingItem(BaseModel):
    category: str = Field(description="The main category: shirt, pants, shoes, jacket, etc.")
    primary_color: str = Field(description="The dominant color.")
    pattern: str = Field(description="e.g., solid, striped, floral, graphic print.")
    style: str = Field(description="e.g., casual, formal, streetwear, athletic.")
    season: str = Field(description="The best season: summer, winter, fall, spring, or all-season.")

class OutfitQuery(BaseModel):
    needed_category: str = Field(description="The category needed to complete the outfit, e.g., 'pants' or 'shoes'.")
    search_prompt: str = Field(description="A descriptive query for vector search, e.g., 'casual light blue relaxed fit denim pants'.")
    styling_reasoning: str = Field(description="Brief explanation of why this complements the base item.")

class StylistPlan(BaseModel):
    recommendations: list[OutfitQuery]

st.title("👗 AI Digital Wardrobe Setup")

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
st.header("✨ AI Stylist Test Zone")

# Let's test with the Navy Shirt you uploaded!
test_base_item = "A casual solid navy blue shirt suitable for summer"
st.write(f"**Base Item:** {test_base_item}")

if st.button("Generate Outfit Plan"):
    with st.spinner("Consulting your AI Stylist..."):
        # 1. Get the plan from Gemini
        plan_json = get_stylist_recommendations(test_base_item)
        plan_dict = json.loads(plan_json)
        
        st.subheader("The Stylist's Plan:")
        st.json(plan_dict) # This will print the raw JSON so we can inspect it!
        
        # 2. Query ChromaDB with the AI's suggestions
        st.subheader("Searching your wardrobe...")
        
        for rec in plan_dict['recommendations']:
            st.write(f"🔍 Searching for: *{rec['search_prompt']}*")
            
            # Query the vector DB WITH Metadata Filtering!
            results = collection.query(
                query_texts=[rec['search_prompt']],
                n_results=1,
                where={"category": rec['needed_category']} 
            )
            
            # Print the results we found
            if results['distances'][0]:
                st.success(f"Match found in database: {results['documents'][0][0]}")
            else:
                st.warning("No close matches found in your wardrobe yet.")
