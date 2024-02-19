import requests
import base64
import time
# URL for the web service
url = "http://127.0.0.1:9880/predictions/groundingdino"
headers = {"Content-Type": "application/json"}

# Input data
with open("test.png", "rb") as f:
    image = f.read()

data = {
        "image": base64.b64encode(image).decode("utf-8"), # base64 encoded image or BytesIO
        "caption": "steel pipe", # text prompt, split by "." for multiple phrases
        "box_threshold": 0.25, # threshold for object detection
        "caption_threshold": 0.25 # threshold for text similarity
        }

# Make the request and display the response

resp = requests.post(url=url, headers=headers, json=data)
outputs = resp.json()