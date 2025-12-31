import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

# ----------------------------
# 1. Select Image from Gallery
# ----------------------------
Tk().withdraw()
image_path = askopenfilename(title="Select Bottle Image",
                             filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

# Load image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image. Please ensure the file path is correct.")
    exit()

original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ----------------------------
# 2. Create Full Bottle Mask (Crucial Adjustment Here)
# ----------------------------
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh = cv2.bitwise_not(thresh) 

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(gray)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 5000: # Adjust based on bottle size
        cv2.drawContours(mask, [cnt], -1, 255, -1)

# Remove cap region: top 15% of bottle
height = mask.shape[0]
mask[:int(0.15*height), :] = 0

# **FIX 1: MASK EROSION REDUCTION**
# We reduce the erosion to ensure the very edge/base of the bottle, where the impurity lies, is not excluded.
kernel = np.ones((2,2), np.uint8) # Changed from (3,3) to (2,2)
mask_inside = cv2.erode(mask, kernel, iterations=1)


# ----------------------------
# 3. Detect minute impurities inside bottle
# ----------------------------
bottle_inside = cv2.bitwise_and(gray, gray, mask=mask_inside)

# Median blur to preserve small particles
blur = cv2.medianBlur(bottle_inside, 3) 

# Adaptive threshold for fine impurities (High Sensitivity)
impurity_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 9, 1)

# **FIX 2: MORPHOLOGY ADJUSTMENT**
# Removed the cv2.morphologyEx(..., cv2.MORPH_OPEN, ...) step entirely. 
# This step can sometimes remove actual small impurities, especially dark ones at the base.
cleaned = impurity_thresh 


# ----------------------------
# 4. Highlight impurities on original image
# ----------------------------
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
detection_count = 0

# **FIX 3: CONTOUR FILTERING**
# We remove any aggressive area filtering here to ensure small particles are kept.
for cnt in contours:
    area = cv2.contourArea(cnt)
    # Using a very low minimum area to capture small particles.
    if area > 0.5: # Lowered the area threshold for extremely small specks
        # Check aspect ratio/solidity to filter out noise if needed, but start with simple area check
        
        # Check if the contour is inside the main mask (optional, but good practice)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if mask_inside[cY, cX] == 255:
                cv2.drawContours(original, [cnt], -1, (0,0,255), 2)  # Thick red highlight
                detection_count += 1

# ----------------------------
# 5. Display results
# ----------------------------
print(f"Total Impurities Detected: {detection_count}")

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Impurities Detected (Cap Ignored, Full Bottle)")
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
