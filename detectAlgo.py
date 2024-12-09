import torch
import cv2

# 1. Load the trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       path=r'C:\Wasutha\03_Junior\DigImage\ChessDetectionV2\yolov5\chess-v2.pt')  # Replace with your model path
model.conf = 0.2  # Set confidence threshold (adjust as needed)
model.iou = 0.9  # Non-Maximum Suppression IoU threshold

# 2. Load the video
video_path = r"C:\Wasutha\03_Junior\DigImage\ChessDetectionV2\yolov5\test-video\2_move_student.mp4"  # Input video path
output_path = r"C:\Wasutha\03_Junior\DigImage\ChessDetectionV2\detectAlgoOutput.avi"  # Output video path (ensure extension)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Video frame rate (frames per second)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 1, (width, height))  # Save at 1 FPS

frame_count = 0
processed_frame = 0

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    xi1, yi1 = max(x1, x1_), max(y1, y1_)
    xi2, yi2 = min(x2, x2_), min(y2, y2_)
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def non_max_suppression_custom(detections, iou_threshold=0.6):
    """Custom NMS to remove overlapping boxes with lower confidence."""
    filtered_boxes = []
    while len(detections) > 0:
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        best_box = detections.pop(0)  # Take the box with highest confidence
        filtered_boxes.append(best_box)

        # Remove boxes with IoU > threshold
        detections = [
            box for box in detections
            if calculate_iou(
                (best_box['xmin'], best_box['ymin'], best_box['xmax'], best_box['ymax']),
                (box['xmin'], box['ymin'], box['xmax'], box['ymax'])
            ) < iou_threshold
        ]

    return filtered_boxes

# 3. Process 1 frame per second
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Only process every Nth frame where N = fps (1 frame per second)
    if frame_count % fps == 0:
        processed_frame += 1
        print(f"Processing frame {frame_count} (Frame {processed_frame} at 1 FPS)")

        # Run YOLO model inference
        results = model(frame)

        # Parse detections
        detections = results.pandas().xyxy[0]  # Get detections as a Pandas DataFrame
        detections_list = [
            {
                'xmin': int(row['xmin']),
                'ymin': int(row['ymin']),
                'xmax': int(row['xmax']),
                'ymax': int(row['ymax']),
                'confidence': float(row['confidence']),
                'name': row['name']
            }
            for _, row in detections.iterrows()
        ]

        # Apply custom NMS
        filtered_detections = non_max_suppression_custom(detections_list, iou_threshold=0.6)

        print(f"Filtered detections for frame {frame_count}:")
        for detection in filtered_detections:
            print(detection)

        # Draw bounding boxes and labels on the frame
        for det in filtered_detections:
            xmin, ymin, xmax, ymax = det['xmin'], det['ymin'], det['xmax'], det['ymax']
            label = f"{det['name']} {det['confidence']:.2f}"
            color = (0, 255, 0) if det['name'] == "chessboard" else (0, 0, 255)  # Green for chessboard, Red for pieces
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the annotated frame to the output video
        out.write(frame)

        # Display the frame (optional, for debugging)
        cv2.imshow('YOLOv5 Chess Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_count += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video with detections (1 FPS) saved to {output_path}")