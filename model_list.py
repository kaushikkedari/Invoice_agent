# import google.generativeai as genai
# import os

# # Configure API key
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Or pass directly

# print("Available Google Generative AI Models (may include embeddings):")
# for m in genai.list_models():
#   # Check if the model supports embedding
#   if 'embedContent' in m.supported_generation_methods:
#     print(m.name)

import google.generativeai as genai
import os
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

models = genai.list_models()

for model in models:
    print(model.name)