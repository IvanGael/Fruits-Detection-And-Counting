import cv2
from ultralytics import YOLO

# Load the YOLOv8 model from the hub
model = YOLO('yolov8n') 

# Function to detect and count fruits in a video frame
def detect_and_count_fruits(frame):
    results = model(frame)  # Perform inference on the frame
    detections = results[0]  # Get detections from the first (and only) item in results
    fruit_count = 0

    # Iterate over all detections and count fruits (e.g., oranges)
    for detection in detections.boxes.data.tolist():  # Convert tensor to a list of detections
        xmin, ymin, xmax, ymax, confidence, class_id = detection[:6]
        class_name = model.names[int(class_id)]  # Access class names from the model
        if class_name == 'orange': 
            fruit_count += 1
            # Draw bounding box around detected fruit
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            # Draw class name and confidence
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    return frame, fruit_count

# Open video file
video_path = 'video.mp4'  
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object to save output video
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # Restart the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Detect and count fruits in the frame
    processed_frame, fruit_count = detect_and_count_fruits(frame)

    # Display the frame with detections
    cv2.putText(processed_frame, f'Count: {fruit_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Frame', processed_frame)

    # Write the processed frame to the output video
    out.write(processed_frame)

    # Press 'q' to exit the video display early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()
