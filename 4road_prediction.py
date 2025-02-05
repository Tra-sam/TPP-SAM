import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


# Function to load road feature points files
def load_csv_data(csv_file):
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract coordinates and labels
    coords = df[['R_I', 'R_J']].values
    labels = np.array([1 if label == 'RP' else 0 for label in df['L']])

    return coords, labels


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


# Path to your Featurepointsfiles
csv_file = './Featurepointsfiles/000000000050.csv'
coords, labels = load_csv_data(csv_file)


# Import the image and prepare for segmentation
image = cv2.imread("./DATA/image/000000000050.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()

# Prepare the SamPredictor model
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"  # or "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

# Use all points from the CSV as inputs
masks, scores, logits = predictor.predict(
    point_coords=coords,
    point_labels=labels,
    multimask_output=True,
)

# Display results
print(masks.shape)  # (number_of_masks) x H x W

# Show multiple masks
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(coords, labels, plt.gca())
    plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()
