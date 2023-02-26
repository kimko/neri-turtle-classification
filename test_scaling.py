import os
import shutil
import cv2
# from train import scale_image
from helpers import resize_image, resize_and_rotate_image


# define the input and output directories
root_directory = "/Volumes/T7/turtles"
output_dir = "test"

# create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# loop through the first 100 files in the input directory
i = 0
for subdir, dirs, files in os.walk(root_directory):
    for filename in files:
        if filename.lower().endswith('.jpg') and not filename.startswith('._'):
            # check if the file is an image
            if filename.endswith(".jpg") or filename.endswith(".png"):
                i += 1
                print(filename)
                if i >= 100:
                    break
                # get the full path to the file
                filepath = os.path.join(subdir, filename)
                # scale the image
                img = cv2.imread(filepath)
                resized_img = resize_and_rotate_image(img)
                # save the scaled image to the output directory
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, resized_img)

print("Done!")
