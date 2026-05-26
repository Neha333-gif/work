import os
from litellm import model_list
# Set GROQ_API_KEY in your environment or .env file before running
# os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"
try:
    print(model_list)
except Exception as e:
    print(e)
