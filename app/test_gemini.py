import os
import google.generativeai as genai

# Disable default credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

# Use only API key
api_key = os.getenv("GEMINI_API_KEY")

# Configure with transport set to rest (not grpc)
genai.configure(
    api_key=api_key,
    transport="rest"  # Force REST API, not gRPC
)

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Hello")
print(response.text)