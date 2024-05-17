import os

# Specify the directory containing the files
directory = r"D:\Users\Horlings\ii_hh\bioinformatics_project\data\masks"

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Check if the file ends with '.tif'
    if filename.endswith('.tif'):
        # Split the filename to get the number
        base, extension = os.path.splitext(filename)
        
        # Create the new filename
        new_filename = f'img{base}_mask{extension}'
        
        # Get the full path for the old and new filenames
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_filepath, new_filepath)

print("Files have been renamed successfully.")
