import requests

# The URL of your API (running locally for now)
url = "http://127.0.0.1:8000/predict"

# Sample data (Ingredients for a specific wine)
payload = {
  "alcohol": 13.2,
  "malic_acid": 1.78,
  "ash": 2.14,
  "alcalinity_of_ash": 11.2,
  "magnesium": 100,
  "total_phenols": 2.65,
  "flavanoids": 2.76,
  "nonflavanoid_phenols": 0.26,
  "proanthocyanins": 1.28,
  "color_intensity": 4.38,
  "hue": 1.05,
  "od280_od315_of_diluted_wines": 3.4,
  "proline": 1050
}

# Send the POST request
print(f"Sending request to {url}...")
response = requests.post(url, json=payload)

# Print the result
if response.status_code == 200:
    print("Success!")
    print("API Response:", response.json())
else:
    print("Error:", response.text)