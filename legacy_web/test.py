import os
from dotenv import load_dotenv
from google import genai
from PIL import Image
from pydantic import BaseModel, Field
from google.genai import types # We need this for the new SDK config

class ClothingItem(BaseModel):
    category: str = Field(description="The main category: shirt, pants, shoes, jacket, etc.")
    primary_color: str = Field(description="The dominant color of the item.")
    pattern: str = Field(description="e.g., solid, striped, floral, graphic print.")
    style: str = Field(description="e.g., casual, formal, streetwear, athletic.")
    season: str = Field(description="The best season for this item: summer, winter, fall, spring, or all-season.")

# 1. Load environment variables from the .env file
load_dotenv()

# 2. Initialize the GenAI Client
# The client automatically looks for the GEMINI_API_KEY environment variable!
client = genai.Client()

# 3. Load the image 
image_path = "example.png"  # Update to your image!
try:
    clothing_image = Image.open(image_path)
except FileNotFoundError:
    print(f"Error: Could not find {image_path}.")
    exit()

print("Extracting structured JSON metadata...")

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        "Analyze this clothing item and extract the requested details.",
        clothing_image
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ClothingItem,
    )
)

print("\n--- Structured JSON Output ---")
print(response.text)