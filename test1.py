import requests
import os

# API endpoint
url = "http://localhost:8000/predict"

# Path to your dataset folder (root folder containing subfolders of images)
image_dir = r"C:/Users/MD Sahil Johan Islam/OneDrive/Desktop/P1/R3/Backend/plantvillage dataset"


# -------------------------------
# Test 1: Single valid image
# -------------------------------
def test_single_image():
    image_name = "0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG"
    image_path = os.path.join(image_dir, image_name)
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_name, f, "image/jpeg")}
            response = requests.post(url, files=files)
            response.raise_for_status()
            print(f"[Test 1] Single Image: {image_name}, Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"[Test 1] Request failed: {e}")
    except FileNotFoundError:
        print(f"[Test 1] File not found: {image_path}")


# -------------------------------
# Test 2: Non-image file
# -------------------------------
def test_non_image_file():
    fake_file_path = os.path.join(image_dir, "fake.txt")
    with open(fake_file_path, "w") as f:
        f.write("This is not an image.")
    try:
        with open(fake_file_path, "rb") as f:
            files = {"file": ("fake.txt", f, "text/plain")}
            response = requests.post(url, files=files)
            print(f"[Test 2] Non-image file: Status {response.status_code}, Response: {response.text}")
    finally:
        os.remove(fake_file_path)


# -------------------------------
# Test 3: Missing file
# -------------------------------
def test_missing_file():
    missing_path = os.path.join(image_dir, "missing.jpg")
    try:
        with open(missing_path, "rb") as f:
            files = {"file": ("missing.jpg", f, "image/jpeg")}
            response = requests.post(url, files=files)
            print(f"[Test 3] Missing file: {response.status_code}")
    except FileNotFoundError:
        print(f"[Test 3] File not found as expected: {missing_path}")


# -------------------------------
# Test 4: Empty file
# -------------------------------
def test_empty_file():
    empty_path = os.path.join(image_dir, "empty.jpg")
    open(empty_path, "wb").close()
    try:
        with open(empty_path, "rb") as f:
            files = {"file": ("empty.jpg", f, "image/jpeg")}
            response = requests.post(url, files=files)
            print(f"[Test 4] Empty file: Status {response.status_code}, Response: {response.text}")
    finally:
        os.remove(empty_path)


# -------------------------------
# Test 5: Multiple valid images (first 3 only)
# -------------------------------
def test_multiple_images():
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))][:3]
    for i, image_name in enumerate(image_files, start=5):
        image_path = os.path.join(image_dir, image_name)
        try:
            with open(image_path, "rb") as f:
                files = {"file": (image_name, f, "image/jpeg")}
                response = requests.post(url, files=files)
                print(f"[Test {i}] Multi-image: {image_name}, Response: {response.json()}")
        except Exception as e:
            print(f"[Test {i}] Error: {e}")


# -------------------------------
# Test 8: Unsupported file type (PDF)
# -------------------------------
def test_pdf_file():
    pdf_path = os.path.join(image_dir, "fake.pdf")
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4 Fake PDF content")
    try:
        with open(pdf_path, "rb") as f:
            files = {"file": ("fake.pdf", f, "application/pdf")}
            response = requests.post(url, files=files)
            print(f"[Test 8] PDF file: Status {response.status_code}, Response: {response.text}")
    finally:
        os.remove(pdf_path)


# -------------------------------
# Test 9: Very large image
# -------------------------------
def test_large_image():
    large_image = os.path.join(image_dir, "large.jpg")
    with open(large_image, "wb") as f:
        f.write(os.urandom(5 * 1024 * 1024))  # 5 MB random data
    try:
        with open(large_image, "rb") as f:
            files = {"file": ("large.jpg", f, "image/jpeg")}
            response = requests.post(url, files=files)
            print(f"[Test 9] Large image: Status {response.status_code}, Response: {response.text}")
    finally:
        os.remove(large_image)


# -------------------------------
# Test 10: No file uploaded
# -------------------------------
def test_no_file():
    response = requests.post(url, files={})
    print(f"[Test 10] No file upload: Status {response.status_code}, Response: {response.text}")


# -------------------------------
# Test All Images in Dataset (recursive + accuracy check)
# -------------------------------
def test_all_images():
    total = 0
    correct = 0
    i = 1
    for root, _, files in os.walk(image_dir):  # walks through subfolders
        label = os.path.basename(root)  # folder name = true label

        for image_name in files:
            image_path = os.path.join(root, image_name)

            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                with open(image_path, "rb") as f:
                    files = {"file": (image_name, f, "image/jpeg")}
                    response = requests.post(url, files=files)
                    response.raise_for_status()
                    result = response.json()

                    predicted = result.get("class", "").lower()
                    true_label = label.lower()

                    # Check correctness
                    if true_label in predicted:
                        correct += 1

                    total += 1
                    print(f"[Image {i}] {image_name} | True: {true_label} | Predicted: {predicted}")
            except Exception as e:
                print(f"[Image {i}] {image_name} → Error: {e}")

            i += 1

    # Summary
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n✅ Dataset Accuracy: {correct}/{total} = {accuracy:.2f}%")
    else:
        print("\n⚠️ No images tested.")


# -------------------------------
# Run all tests
# -------------------------------
if __name__ == "__main__":
    test_single_image()
    test_non_image_file()
    test_missing_file()
    test_empty_file()
    test_multiple_images()
    test_pdf_file()
    test_large_image()
    test_no_file()
    test_all_images()
