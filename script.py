import os
import uuid
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from ultralytics import YOLO
import random
from uuid import uuid4
import os


class_list= ['-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
model = YOLO("best.pt") 
conf = 0.85
img_resolution = 640

def detect_v8(image_path):
    image_path = os.path.normpath(image_path)
    if image_path:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
    else:
        print("Image path not provided.")
        return
    
    # Get the dimensions of the image to determine orientation
    img_width, img_height = img.size
    
    # Check orientation (horizontal or vertical)
    if img_width >= img_height:
        orientation = 'horizontal'
        print("Orientation: Horizontal")
    else:
        orientation = 'vertical'
        print("Orientation: Vertical")
    
    # Perform detection using YOLO
    results = model.predict(source=img_np,
                            imgsz=img_resolution)
    
    tensor_list = results[0].boxes.data
    detection = tensor_list.tolist()
    total_products = len(detection)
    print(f"Total detected characters/numbers: {total_products}")
    
    # Store the detected characters along with their positions
    detections = []
    
    for det in detection:
        confidence = det[4]
        x, y, w, h, _, cls = [int(d) for d in det]
        detected_class = class_list[cls]
        detections.append((x, y, detected_class, confidence))  # Store x-coordinate, y-coordinate, class, and confidence

        # Draw bounding boxes and text on the image
        cv2.rectangle(img_np, (x, y), (w, h), (255, 0, 0), 1)
        if detected_class in class_list:
            text_y_coordinate = y + 13  # Move the text down by 13 pixels
        else:
            text_y_coordinate = y - 10  # Keep original position for other classes
        
        cv2.putText(img_np, f'{str(detected_class)}:{str(round(confidence, 2))}', 
                    (x, text_y_coordinate), 2, 0.6, (0, 0, 255), 1)
    
    # Sort detections based on orientation
    if orientation == 'horizontal':
        # Sort by the x-coordinate (left to right)
        detections_sorted = sorted(detections, key=lambda d: d[0])
        detected_text = ''.join([d[2] for d in detections_sorted])  # Concatenate the detected characters/numbers
        print(f"Detected text (left to right): {detected_text}")
    else:
        print("For vertical text, choose reading direction:")
        print("Option 1: Top to Bottom")
        print("Option 2: Bottom to Top")
        option = input("Enter Option (1 or 2): ")
        if option == "2":
            # Sort by the y-coordinate (bottom to top)
            detections_sorted = sorted(detections, key=lambda d: -d[1])  # Sort y in descending order (bottom to top)
            detected_text = ''.join([d[2] for d in detections_sorted])  # Concatenate the detected characters/numbers
            print(f"Detected text (bottom to top): {detected_text}")
        elif option == "1":
            # Sort by the y-coordinate (top to bottom)
            detections_sorted = sorted(detections, key=lambda d: d[1])  # Sort y in ascending order (top to bottom)
            detected_text = ''.join([d[2] for d in detections_sorted])  # Concatenate the detected characters/numbers
            print(f"Detected text (top to bottom): {detected_text}")
        
    # Display the image with bounding boxes and labels
    img_pil = Image.fromarray(np.uint8(img_np))
    output_filename = f"output_detected_image_{str(uuid4())}.jpg"
    output_dir = os.path.normpath(os.path.join(os.getcwd(), output_filename))
    img_pil.save(output_dir)

# The folder path and images will be handled in the next cell

image_path = "D:/text/New_folder/m.jpg"
detect_v8(image_path)

