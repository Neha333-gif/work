import json
import urllib.request
text = "Congratulations! You've been selected to receive a $1,000 gift card. Click http://bit.ly/claim-now or call 1-800-555-0100 within 24 hours to redeem. Reply YES to confirm your prize."
data = json.dumps({"email_text": text}).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:8000/predict', data=data, headers={'Content-Type':'application/json'})
resp = urllib.request.urlopen(req, timeout=15)
print(resp.read().decode('utf-8'))
