# import requests

# url = "http://localhost:8000/predict"
# file_path = r"C:\Users\MD Sahil Johan Islam\OneDrive\Desktop\P1\R3\Backend\plantvillage dataset\color\Apple___Apple_scab\0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG"

# try:
#     with open(file_path, "rb") as f:
#         files = {"file": ("image.jpg", f, "image/jpeg")}
#         response = requests.post(url, files=files)
#         response.raise_for_status()
#         print(response.json())
# except requests.exceptions.RequestException as e:
#     print(f"Request failed: {e}")
# except FileNotFoundError:
#     print(f"File not found: {file_path}")

import requests
import os

url = "http://localhost:8000/predict"
image_dir = r"C:\Users\MD Sahil Johan Islam\OneDrive\Desktop\P1\R3\Backend\plantvillage dataset\color\Apple___Apple_scab"

def test_single_image():
    image_name = "0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG"
    image_path = os.path.join(image_dir, image_name)
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_name, f, "image/jpeg")}
            response = requests.post(url, files=files)
            response.raise_for_status()
            print(f"Single Image Test: {image_name}, Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {image_name}: {e}")
    except FileNotFoundError:
        print(f"File not found: {image_path}")

def test_non_image_file():
    fake_file_path = os.path.join(image_dir, "fake.txt")
    with open(fake_file_path, "w") as f:
        f.write("This is not an image.")
    try:
        with open(fake_file_path, "rb") as f:
            files = {"file": ("fake.txt", f, "text/plain")}
            response = requests.post(url, files=files)
            print(f"Non-image file test: Status {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed for fake.txt: {e}")
    finally:
        os.remove(fake_file_path)

def test_missing_file():
    missing_path = os.path.join(image_dir, "missing.jpg")
    try:
        with open(missing_path, "rb") as f:
            files = {"file": ("missing.jpg", f, "image/jpeg")}
            response = requests.post(url, files=files)
            print(f"Missing file test: {response.status_code}")
    except FileNotFoundError:
        print(f"File not found as expected: {missing_path}")

def test_empty_file():
    empty_path = os.path.join(image_dir, "empty.jpg")
    open(empty_path, "wb").close()
    try:
        with open(empty_path, "rb") as f:
            files = {"file": ("empty.jpg", f, "image/jpeg")}
            response = requests.post(url, files=files)
            print(f"Empty file test: Status {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed for empty.jpg: {e}")
    finally:
        os.remove(empty_path)

def test_multiple_images():
    image_files = os.listdir(image_dir)[:3]
    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        try:
            with open(image_path, "rb") as f:
                files = {"file": (image_name, f, "image/jpeg")}
                response = requests.post(url, files=files)
                print(f"Multiple Images Test: {image_name}, Response: {response.json()}")
        except Exception as e:
            print(f"Error for {image_name}: {e}")

if __name__ == "__main__":
    test_single_image()
    test_non_image_file()
    test_missing_file()
    test_empty_file()
    test_multiple_images()