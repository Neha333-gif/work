import requests
import json

BASE_URL = "http://localhost:8000"

# Test 1: Health check
print("=== Testing Health Endpoint ===")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
except Exception as e:
    print(f"Error: {e}\n")

# Test 2: Single prediction
print("=== Testing Prediction Endpoint ===")
data = {
    "gender": "Male",
    "senior_citizen": 0,
    "tenure": 24,
    "monthly_charges": 65.5,
    "total_charges": 1570.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check"
}

try:
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Churn Prediction: {result.get('churn_prediction')}")
    print(f"Churn Probability: {result.get('churn_probability')}%")
    print(f"Risk Level: {result.get('risk_level')}")
    print(f"Model Confidence: {result.get('model_confidence')}%")
    print()
except Exception as e:
    print(f"Error: {e}\n")

# Test 3: Get info
print("=== Testing Info Endpoint ===")
try:
    response = requests.get(f"{BASE_URL}/info")
    print(f"Status: {response.status_code}")
    print(f"API Title: {response.json().get('title')}")
    print(f"Version: {response.json().get('version')}")
    print()
except Exception as e:
    print(f"Error: {e}\n")

# Test 4: Get metrics
print("=== Testing Metrics Endpoint ===")
try:
    response = requests.get(f"{BASE_URL}/results/metrics")
    print(f"Status: {response.status_code}")
    metrics = response.json()
    print(f"Report preview (first 200 chars):")
    print(metrics.get('text_report', '')[:200])
    print(f"\nVisualizations available:")
    for name in metrics.get('visualizations', {}).keys():
        print(f"  - {name}")
    print()
except Exception as e:
    print(f"Error: {e}\n")

print("=== All API Tests Completed ===")
