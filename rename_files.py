import os

def crawl_and_prepend(root_dir):
    names = []
    # Crawl through all subdirectories and files in root_dir
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # Prepend file path to file name, but remove the root directory from the beginning of the file path, and replace '/' and ' ' with '_'
            file_path = os.path.join(subdir, file)
            prepended_name = file_path.replace(root_dir, '', 1).replace('/', '_').replace(' ', '_').replace('Turtle_Pics', '_')
            prepended_name = prepended_name[1:]
            # add prepended_name to names list
            names.append(prepended_name)
            print(prepended_name)
    return names

def export_names_to_file(names):
    with open("file_names.txt", "w") as f:
        for name in names:
            f.write(f"{name}\n")




# Test the function
crawl_and_prepend("/path/to/root/directory")

# call crawl_and_prepend() in __main__ with argument
if __name__ == "__main__":
    import sys
    names = crawl_and_prepend(sys.argv[1])
    export_names_to_file(names)

# Run the script with the following command:
# python rename_files.py /path/to/root/directory


