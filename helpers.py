import os
import cv2
import numpy as np

def resize_and_rotate_image(img, a=512, b=384, filename=""):
    '''
    Resize the image with new dimensions a and b, where a is smaller than b.
    Apply a to the smaller existing dimension and b to the larger existing dimension.
    Rotate the image so that the higher dimension becomes the width.
    '''
    # Get the current dimensions of the image
    h, w = img.shape[:2]

    # Determine which dimension to apply a and b to
    if h < w:
        new_h = a
        new_w = b
    else:
        new_h = b
        new_w = a

    # Calculate the scaling factors
    h_scale = new_h / h
    w_scale = new_w / w

    # Apply the scaling factors to the image
    img_resized = cv2.resize(img, (int(h * h_scale), int(w * w_scale)))
    # print(f"{int(h * h_scale):5d} {int(w * w_scale):5d} {img_resized.shape[:2]} {filename}")
    # # Rotate the image so that the higher dimension becomes the width
    if w < h:
        img_rotated = cv2.rotate(img_resized, cv2.ROTATE_90_CLOCKWISE)
    else:
        img_rotated = img_resized

    # ensure that the final image is exactly a x b
    img_rotated = cv2.resize(img_rotated, (a, b))
    return img_rotated


def resize_image(img, scale_percent=50):
    # Get the current height and width
    height, width = img.shape[:2]

    # Calculate the new height and width based on the aspect ratio
    new_height = int(height * scale_percent / 100)
    new_width = int(width * scale_percent / 100)

    # Resize the image
    # resized_img = cv2.resize(img, (new_width, new_height))
    resized_img = cv2.resize(img, (1024, 768))

    return resized_img

def find_common_dimension(root_dir):
    """
    Find the smallest common width and height amongst all images in the subdirectories of root_dir.

    Args:
        root_dir (str): Root directory to start searching for images.

    Returns:
        tuple: A tuple of integers representing the smallest common width and height amongst all images.

    Raises:
        ValueError: If no images are found in the subdirectories.
    """
    # Initialize the smallest width and height to infinity
    smallest_width = 100000
    smallest_height = 100000

    # Traverse through all subdirectories to find images
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith('.jpg') and not file.startswith('._'):
                # Load the image and get its dimensions
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)
                height, width, _ = img.shape

                print(f"{width:5d} {height:5d} {file}")
                # Update the smallest width and height if necessary
                if width < smallest_width:
                    smallest_width = width
                    print(f"{smallest_width:5d} {smallest_height:5d} {width:5d} {height:5d} {file}")
                if height < smallest_height:
                    smallest_height = height
                    print(f"{smallest_width:5d} {smallest_height:5d} {width:5d} {height:5d} {file}")
                #  print(f"{smallest_width:5d} {smallest_height:5d} {width:5d} {height:5d} {file}")

    # Check if any images were found
    if smallest_width == float('inf') or smallest_height == float('inf'):
        raise ValueError('No images found in the subdirectories.')

    # Return the smallest common width and height
    return smallest_width, smallest_height

def scale_image(path):
    '''
    scale the images such that only the turtle is visible without any distracting background, 
    use a technique called object detection. 
    Object detection involves identifying the region of interest (ROI) in an image and cropping 
    it to create a new image with just the desired object.
    '''
    # Load image
    # Load image
    img = cv2.imread(path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to isolate the turtle from the background
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour as the turtle
    cnt = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the turtle contour
    x, y, w, h = cv2.boundingRect(cnt)

    # Crop the image to the bounding rectangle
    crop_img = img[y:y+h, x:x+w]

    # Resize the cropped image to the desired size
    resized_img = cv2.resize(crop_img, (100, 100))
    return resized_img

def remove_carapace_prefix(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if 'CARAPACE_' in filename:
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace('CARAPACE_', '')
                new_path = os.path.join(dirpath, new_filename)
                os.rename(old_path, new_path)

# remove_carapace_prefix("/Volumes/T7/turtles")
# print(find_common_dimension("/Volumes/T7/turtles_training_data"))