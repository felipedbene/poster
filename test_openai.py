import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# Load key from env
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "❌ OPENAI_API_KEY not set"

# Use the same client setup
client = OpenAI(
    api_key=api_key,
    http_client=httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0, read=10.0, write=10.0))
)

prompt = "Summarize this headline in one sentence: 'Robot see, robot do: System learns after watching how-to videos'"

print("📡 Sending request...")
try:
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    print("✅ Response:", response.choices[0].message.content)
except Exception as e:
    print("❌ OpenAI call failed:", e)