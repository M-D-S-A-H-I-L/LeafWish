import zipfile
import pathlib
import os

# Print current working directory to confirm where the script is running
print(f"Current working directory: {os.getcwd()}")

# List all files in the current directory to verify ZIP file presence
print("Files in current directory:")
for file in os.listdir("."):
    print(f"  - {file}")

# Define relative path for the ZIP file
zip_path = pathlib.Path("plantvillage-dataset.zip")
# Define extraction path (subfolder named 'extracted')
extract_path = pathlib.Path("extracted")

# Print the expected ZIP file path
print(f"Looking for ZIP file at: {zip_path.resolve()}")

# Check if the ZIP file exists
if not zip_path.exists():
    print(f"Error: The ZIP file '{zip_path}' was not found in {os.getcwd()}.")
    print("Please ensure 'plantvillage-dataset.zip' is in the same folder as this script.")
    print("If the file has a different name, update the 'zip_path' variable to match.")
    exit()

# Ensure the extraction directory exists
extract_path.mkdir(parents=True, exist_ok=True)

try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if zip_ref.testzip() is not None:
            print("Error: The ZIP file is corrupted.")
        else:
            zip_ref.extractall(extract_path)
            print(f"Successfully extracted ZIP file to {extract_path.resolve()}")
except PermissionError:
    print(f"Error: Permission denied when accessing {extract_path}.")
except zipfile.BadZipFile:
    print("Error: The file is not a valid ZIP file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")