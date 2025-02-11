# Import statements
import numpy as np
import argparse
import cv2
import os
from PIL import Image  # Importing Pillow for image resizing

# Paths to load the model
DIR = r"D:\Colorization"
PROTOTXT = os.path.join(DIR, r"D:\Colorization\models/colorize.prototext")
POINTS = os.path.join(DIR, r"D:\Colorization\models/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"D:\Colorization\models/release.caffemodel")

# Argparser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
help="path to input black and white image")
args = vars(ap.parse_args())

# Load the Model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load and Resize the Input Image
print("Loading and resizing the image")
image = cv2.imread(args["image"])

# Resize the image using PIL
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to PIL format
pil_resized = pil_image.resize((600, 600))  # Resize to 600x600 pixels
image = cv2.cvtColor(np.array(pil_resized), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV format (BGR)

# Convert the resized image to LAB color space
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# Resize LAB image to 224x224 (for the colorization network input)
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# Colorize the image
print("Colorizing the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize the ab channels back to the original size (600x600)
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# Merge the L channel with the colorized ab channels
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# Convert LAB back to BGR
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

# Display the original and colorized images
cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)

cv2.waitKey(0)
cv2.destroyAllWindows()
