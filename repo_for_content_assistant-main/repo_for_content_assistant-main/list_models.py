import google.generativeai as genai
import os

API_KEY = "AIzaSyAYfXkDm3036-HZpl676_WPAY_6XE73fw4"
genai.configure(api_key=API_KEY)

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
