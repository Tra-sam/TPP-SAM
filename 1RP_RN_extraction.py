import cv2
import numpy as np
import os

# Extract the positions of white pixels(255) from the image
#recoding part of the TPP-SAM
def extract_white_pixels(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to convert the grayscale image to binary
    ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # Get the positions of white pixels
    white_pixels = np.argwhere(thresh == 255)
    return white_pixels.tolist()


#RP extraction
#Input folder containing the RpRasterMap
input_folder = "./DATA/Rprastermap"

# Output folder for saving the text files
output_folder ="./DATA/Rptxt"

#RN extraction
# # Input folder containing the RnRasterMap
# input_folder = "./DATA/Rnrastermap"
# # Output folder for saving the text files
# output_folder ="./DATA/Rntxt"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for image_file in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    # Extract the positions of white pixels
    white_pixels = extract_white_pixels(image)
    file_name = os.path.splitext(image_file)[0]
    # Save the white pixel positions to a text file
    output_file_path = os.path.join(output_folder, f"{file_name}.txt")
    with open(output_file_path, "w") as file:
        for pixel in white_pixels:
            file.write(f"{pixel[1]},{pixel[0]}\n")
