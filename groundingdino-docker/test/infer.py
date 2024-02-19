import requests
import base64
import time
import cv2
# URL for the web service
url = "http://192.168.0.37:9880/predictions/groundingdino"
headers = {"Content-Type": "application/json"}

# Input data
image = cv2.imread("bird.webp")
_, img_encoded = cv2.imencode('.jpg', image)
img_bytes = img_encoded.tobytes()
data = {
        "image": base64.b64encode(img_bytes).decode("utf-8"), # base64 encoded image or BytesIO
        "caption": "bird.plant", # text prompt, split by "." for multiple phrases
        "box_threshold": 0.25, # threshold for object detection
        "caption_threshold": 0.25 # threshold for text similarity
        }

# Make the request and display the response

resp = requests.post(url=url, headers=headers, json=data)
outputs = resp.json()
print(outputs)

for n in range(len(outputs["boxes"])):
    box = outputs["boxes"][n]
    box = [int(x) for x in box]
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    # phrase
    label = outputs["phrases"][n]
    score = outputs["scores"][n]
    ptext = (box[0], box[1])
    title = f"{label}:{score:.4f}"
    cv2.putText(image, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite("test.jpg", image)
