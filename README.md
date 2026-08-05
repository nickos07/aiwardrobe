# AI Digital Wardrobe & Personal Stylist

An intelligent, multimodal recommender system that digitizes your physical wardrobe and uses Generative AI to curate personalized outfits.

## The Elevator Pitch
This project bridges the gap between Computer Vision and Recommender Systems. Instead of relying on manual data entry, users simply upload photos of their clothing. The system utilizes **Vision-Language Models (VLMs)** to extract structured semantic metadata, embeds those concepts into a **Vector Database**, and acts as an autonomous stylist by performing **Hybrid Semantic Searches** to build cohesive outfits.

## System Architecture

1. **Multimodal Ingestion Pipeline:** Uses Google's Gemini Vision API to analyze raw pixels and extract clothing features (color, pattern, style, season).
2. **Deterministic Data Enforcement:** Uses `Pydantic` and strict JSON schemas to prevent LLM hallucination and ensure strict hierarchical taxonomies (`macro-category` vs. `item_type`).
3. **Hybrid Retrieval Engine:** Combines **ChromaDB** semantic vector search (Cosine Similarity) with strict metadata filtering to retrieve the mathematically closest matches in the user's wardrobe.
4. **State-Managed Frontend:** Built with Streamlit, utilizing rigorous `st.session_state` memoization to cache expensive inference calls and prevent declarative UI re-render loops.

## 🛠️ Tech Stack
* **Frontend:** Python, Streamlit
* **AI / ML:** Google Gemini 2.5 Flash API (Multimodal VLM)
* **Data Validation:** Pydantic
* **Vector Database:** ChromaDB
* **Image Processing:** Pillow (PIL)
* **Cloud Deployment:** Streamlit Community Cloud

## Engineering Challenges & Solutions (The "Internship Flex")

Building this pipeline required solving several complex infrastructure and ML alignment problems:

* **Managing Taxonomy Drift:** Initial generative outputs suffered from a "vocabulary gap" (e.g., the AI searching for "camel trousers" while the database held "cream pants"). I solved this by implementing a two-tier hierarchical taxonomy, separating deterministic filtering tags (`category`) from semantic embedding descriptions (`item_type`), and hard-prompting the LLM to align its search outputs with the database ingestion format.
* **API Throttling & Geo-Restrictions:** Encountered free-tier rate limits. I engineered a resilient frontend architecture using **Session State Memoization** to cache successful API responses, completely decoupling the inference logic from Streamlit's re-render cycles to protect token buckets.
* **Transient Dependency Conflicts in the Cloud:** During cloud deployment, the serverless container failed due to a `GZipResponder` middleware conflict within ChromaDB's ASGI server stack. I diagnosed the underlying tree mismatch and explicitly pinned `starlette` and `fastapi` in the build pipeline to ensure deterministic container compilation.

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/nickos07/aiwardrobe](https://github.com/nickos07/aiwardrobe)
   cd aiwardrobe