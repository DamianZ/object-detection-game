import cv2
import tensorflow as tf
import numpy as np
import random

# Load pre-trained model
model = tf.saved_model.load('model/ssd_mobilenet_v2_coco_2018_03_29.pb')

# Initialize camera
cap = cv2.VideoCapture(0)

# List of objects to choose from
objects = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def detect_objects(frame):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

def main():
    while True:
        # Randomly choose an object
        chosen_object = random.choice(objects)
        print(f"Show a {chosen_object}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects in the frame
            detections = detect_objects(frame)

            # Check if the chosen object is detected
            for detection in detections['detection_classes'][0]:
                if objects[int(detection)] == chosen_object:
                    print(f"{chosen_object} detected!")
                    break

            # Display the frame
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
