import cv2
img = cv2.imread(r"C:\Users\MD Sahil Johan Islam\OneDrive\Desktop\P1\R3\Backend\plantvillage dataset\color\Apple___Apple_scab\0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
if img is None:
    print("Failed to load image")
else:
    print("Image loaded successfully")