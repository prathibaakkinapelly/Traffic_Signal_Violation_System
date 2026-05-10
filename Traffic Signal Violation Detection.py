import cv2
import os
from ultralytics import YOLO, solutions
import numpy as np
from datetime import datetime
import csv

# Change working directory (adjust to your repo path)
os.chdir(r"c:\Users\prath\OneDrive\Desktop\DL_Proj\Traffic_signal_violation")

# Create a folder to save violation images
if not os.path.exists("violations"):
    os.makedirs("violations")

# CSV log file setup
log_file = "violations/violations_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Vehicle_Type", "Confidence", "Image_File"])

RedLight = np.array([[998, 125],[998, 155],[972, 152],[970, 127]])
GreenLight = np.array([[971, 200],[996, 200],[1001, 228],[971, 230]])
ROI = np.array([[910, 372],[388, 365],[338, 428],[917, 441]])

model = YOLO("yolov8m.pt")
coco = model.model.names

TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]

def is_region_light(image, polygon, brightness_threshold=128):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_image)
    cv2.fillPoly(mask, [np.array(polygon)], 255)
    roi = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    mean_brightness = cv2.mean(roi, mask=mask)[0]
    return mean_brightness > brightness_threshold

def draw_text_with_background(frame, text, position, font, scale, text_color, background_color, border_color, thickness=2, padding=5):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  background_color, 
                  cv2.FILLED)
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  border_color, 
                  thickness)
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)

cap = cv2.VideoCapture("tr.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video ended.")
        break
    else:
        frame = cv2.resize(frame, (1100, 700))
        cv2.polylines(frame, [RedLight], True, [0, 0, 255], 1)
        cv2.polylines(frame, [GreenLight], True, [0, 255, 0], 1)
        cv2.polylines(frame, [ROI], True, [255, 0, 0], 2)
        
        results = model.predict(frame, conf=0.75)
        for result in results:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            classes = result.boxes.cls
            
            for box, conf, cls in zip(boxes, confs, classes):
                if coco[int(cls)] in TargetLabels:
                    x, y, w, h = map(int, box)
                    cv2.rectangle(frame, (x, y), (w, h), [0, 255, 0], 2)
                    draw_text_with_background(frame, 
                                      f"{coco[int(cls)].capitalize()}, conf:{(conf)*100:0.2f}%", 
                                      (x, y - 10), 
                                      cv2.FONT_HERSHEY_COMPLEX, 
                                      0.6, 
                                      (255, 255, 255),
                                      (0, 0, 0),
                                      (0, 0, 255))

                # Check for red light violation
                if is_region_light(frame, RedLight):
                    if cv2.pointPolygonTest(ROI, (x, y), False) >= 0 or cv2.pointPolygonTest(ROI, (w, h), False) >= 0:
                        draw_text_with_background(frame, 
                                      f"The {coco[int(cls)].capitalize()} violated the traffic signal.", 
                                      (10, 30), 
                                      cv2.FONT_HERSHEY_COMPLEX, 
                                      0.6, 
                                      (255, 255, 255),
                                      (0, 0, 0),
                                      (0, 0, 255))

                        cv2.polylines(frame, [ROI], True, [0, 0, 255], 2)
                        cv2.rectangle(frame, (x, y), (w, h), [0, 0, 255], 2)

                        # === Save violation frame ===
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        img_name = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        img_path = f"violations/{img_name}"
                        cv2.imwrite(img_path, frame)
                        print(f"[SAVED] Violation snapshot -> {img_path}")

                        # === Log violation details to CSV ===
                        with open(log_file, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([timestamp, coco[int(cls)], f"{conf*100:.2f}%", img_name])
    
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:
            break
        
cap.release()
cv2.destroyAllWindows()
