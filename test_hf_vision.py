import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# Initialize the HF Client
client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))

# We will use Qwen2-VL-7B as our test model
model_id = "Qwen/Qwen2-VL-7B-Instruct"

# A strict prompt forcing JSON
prompt = """
Analyze this clothing item. You must return ONLY a raw, valid JSON object with no markdown formatting, no backticks, and no conversational text. 
Use this exact schema:
{
  "category": "top, bottom, or shoes",
  "primary_color": "color",
  "pattern": "pattern",
  "style": "style",
  "season": "season"
}
"""

# Note: The free HF Inference API usually requires image URLs rather than local PIL files for VLMs
image_url = "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?q=80&w=500" # A white t-shirt

print(f"Querying {model_id} via Hugging Face...")

try:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=200
    )
    
    output_text = response.choices[0].message.content
    print("\n--- Raw Output ---")
    print(output_text)
    
    # Let's see if it parses successfully!
    parsed_json = json.loads(output_text)
    print("\n✅ Success! Parsed JSON:")
    print(parsed_json)

except Exception as e:
    print(f"\n❌ Error: {e}")