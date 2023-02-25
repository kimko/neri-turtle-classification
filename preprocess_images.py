'''
This script will pre-process the images in the directory tree.
1) find_bottom_folders and crete a key-value pair of the folder path and a running integer number

'''


import csv
import os

def find_bottom_folders(root_dir):
    """
    Recursively traverse the directory tree starting at root_dir, and identify
    the bottom-most folders. Returns a list of tuples, where each tuple contains
    a running integer number and the bottom-most folder path
    """
    folder_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirnames: # if there are no more directories in this directory
            folder_paths.append(dirpath)
    folder_paths.sort() # ensure consistent ordering of paths
    folder_list = [(i, path) for i, path in enumerate(folder_paths)]
    return folder_list

def prepend_turtle_id(folder_list):
    """
    Iterate over each tuple in folder_list, prepending the turtle_id to each file
    in the corresponding directory, and renaming the file with the updated name.
    """
    for turtle_id, folder_path in folder_list:
        for filename in os.listdir(folder_path):
            old_filepath = os.path.join(folder_path, filename)
            new_filename = f"{turtle_id:04}_{filename}"
            new_filepath = os.path.join(folder_path, new_filename)
            os.rename(old_filepath, new_filepath)

def export_folder_list(folder_list, filename):
    """
    Export the folder_list as a CSV file with headers "turtle_id" and "path".
    """
    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["turtle_id", "path"])
        for turtle_id, path in folder_list:
            writer.writerow([f"{turtle_id:04}", path])


# Run the script with the following command:
# python preprocess_images.py /path/to/root/directory
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
        turtle_folders = find_bottom_folders(root_dir)

turtle_folders = find_bottom_folders(
    "/Volumes/T7/turtles"
)