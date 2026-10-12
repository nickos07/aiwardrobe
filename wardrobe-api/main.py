import io
import json
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from google import genai
import chromadb
import uuid
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Literal

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="AI Wardrobe API")

os.makedirs("static/images", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

chroma_client = chromadb.PersistentClient(path="./api_wardrobe_db")
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

class OutfitQuery(BaseModel):
    needed_category: str = Field(description="e.g., 'top', 'bottom', or 'shoes'.")
    selected_item_id: str = Field(description="The EXACT ID of the item chosen from the provided wardrobe inventory.")
    styling_reasoning: str = Field(description="Brief explanation of why this complements the base item.")

# --- NEW SCHEMA ---
class OutfitOption(BaseModel):
    outfit_name: str = Field(description="A catchy name for this look (e.g., 'Weekend Casual' or 'Elevated Evening').")
    outfit_vibe: str = Field(description="A short sentence describing the overall aesthetic.")
    items: list[OutfitQuery] 

class StylistPlan(BaseModel):
    outfits: list[OutfitOption]

@app.post("/upload")
async def upload_clothing_item(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    unique_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1]
    saved_filename = f"{unique_id}.{file_extension}"
    save_path = f"static/images/{saved_filename}"

    image.save(save_path)

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=["Analyze this clothing item and extract the requested details.", image],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClothingItem,
        ),
    )

    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        extracted_metadata = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
    elif response.text:
        extracted_metadata = json.loads(response.text)
    else:
        extracted_metadata = {}

    # 3. Create the Embeddable String
    summary_string = f"A {extracted_metadata['style']} {extracted_metadata['pattern']} {extracted_metadata['primary_color']} {extracted_metadata['item_type']} suitable for {extracted_metadata['season']}"
    
    # 4. Construct the public image URL that the mobile app will use
    image_url = f"http://127.0.0.1:8000/static/images/{saved_filename}"
    
    # 5. Insert into ChromaDB
    collection.add(
        documents=[summary_string],
        metadatas=[{"image_url": image_url, **extracted_metadata}], 
        ids=[unique_id]
    )

    return {
        "status": "success",
        "item_id": unique_id,
        "image_url": image_url,
        "metadata": extracted_metadata
    }

@app.get("/generate-outfit/{item_id}")
async def generate_outfit(item_id: str):
    # 1. Look up the base item in ChromaDB using the unique ID
    results = collection.get(ids=[item_id])

    if not results['documents']:
        raise HTTPException(status_code=404, detail="Clothing item not found in Wardrobe.")

    base_item_summary = results['documents'][0]
    base_item_metadata = results['metadatas'][0]

    all_items = collection.get()

    inventory_catalog = ""
    for i in range(len(all_items['ids'])):
        current_id = all_items['ids'][i]
        if current_id != item_id:
            doc_description = all_items['documents'][i]
            inventory_catalog += f"- ID: {current_id} | Description: {doc_description}\n"

    print("--- INVENTORY CATALOG ---")
    print(inventory_catalog)

    # 2. Ask Gemini to generate 2 distinct outfit combinations around the base item
    prompt = f"""
    The user is building an outfit around this base item: '{base_item_summary}'.
    Act as a high-end personal fashion stylist. 
    
    Here is the user's available wardrobe inventory:
    {inventory_catalog}
    
    CRITICAL RULES:
    1. Create EXACTLY 2 distinct outfit combinations.
    2. You MUST ONLY select items from the provided wardrobe inventory above. Do not invent items.
    3. Return the EXACT 'ID' of the item you select in the 'selected_item_id' field.
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StylistPlan,
        ),
    )

    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        stylist_plan = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
    elif response.text:
        stylist_plan = json.loads(response.text)
    else:
        stylist_plan = {"outfits": []}

    # 3. For each recommended item, query ChromaDB for the best matching wardrobe item in that category
    compiled_outfits = []
    for outfit in stylist_plan.get("outfits", []):
        compiled_items = []
        for rec in outfit.get("items", []):
            needed_category = rec.get("needed_category")
            selected_item_id = rec.get("selected_item_id")

            match_results = collection.get(ids=[selected_item_id])

            image_url = None
            if match_results.get("metadatas") and len(match_results["metadatas"]) > 0:
                first_match = match_results["metadatas"][0]
                image_url = first_match.get("image_url")

            compiled_items.append({
                "needed_category": needed_category,
                "selected_item_id": selected_item_id,
                "styling_reasoning": rec.get("styling_reasoning"),
                "image_url": image_url
            })

        compiled_outfits.append({
            "outfit_name": outfit.get("outfit_name"),
            "outfit_vibe": outfit.get("outfit_vibe"),
            "items": compiled_items
        })

    # 4. Return the fully compiled outfit data to the phone
    return {
        "status": "success",
        "base_item_url": base_item_metadata['image_url'],
        "outfits": compiled_outfits
    }
