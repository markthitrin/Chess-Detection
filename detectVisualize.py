import chess
import chess.svg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.patches import Rectangle

# Sample function to calculate IoU between two bounding boxes
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2

    # Calculate the intersection area
    inter_area = max(0, min(x2, xx2) - max(x1, xx1)) * max(0, min(y2, yy2) - max(y1, yy1))
    
    # Calculate the areas of the boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xx2 - xx1) * (yy2 - yy1)
    
    # Calculate the union area
    union_area = box1_area + box2_area - inter_area
    
    # Return IoU
    return inter_area / union_area if union_area != 0 else 0

# Chessboard setup using chess.svg
board = chess.Board()

# Create an SVG image of the chessboard
chessboard_svg = chess.svg.board(board)

# Sample bounding boxes (format: (x1, y1, x2, y2))
boxes = [
    (50, 50, 150, 150),  # First box
    (120, 120, 180, 180),  # Second box
    (200, 200, 300, 300)   # Third box
]

# IoU threshold
iou_threshold = 0.6

# Filter boxes based on IoU
filtered_boxes = []
for i, box1 in enumerate(boxes):
    add_box = True
    for j, box2 in enumerate(boxes):
        if i != j and iou(box1, box2) > iou_threshold:
            add_box = False
            break
    if add_box:
        filtered_boxes.append(box1)

# Create a figure to display the image
fig, ax = plt.subplots()

# Render the chessboard SVG as an image
img = chess.svg.board(board)
ax.imshow(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))

# Overlay the bounding boxes
for box in filtered_boxes:
    rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

# Show the image with bounding boxes
plt.show()
