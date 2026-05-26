import requests
import json

url = "http://127.0.0.1:8000/command"
data = {"command": "turn on the lights"}
response = requests.post(url, json=data)
print(response.status_code)
print(response.json())
