import cv2
import numpy as np
from tkinter import Tk, filedialog

# Function to create a file dialog to select an image
def select_image():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path

# Load the pre-trained Haar Cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Select an image file
image_path = select_image()
if image_path:
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or cannot be loaded.")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect eyes in the image
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

        # Example transformation matrix for mapping eye coordinates to screen coordinates
        # This is a simple scaling matrix; in real applications, this would be more complex.
        transformation_matrix = np.array([
            [1.2, 0, 0],
            [0, 1.2, 0],
            [0, 0, 1]
        ])

        # Function to transform coordinates using the transformation matrix
        def transform_coordinates(coord, matrix):
            coord = np.append(coord, 1)  # Convert to homogeneous coordinates
            transformed_coord = np.dot(matrix, coord)
            return transformed_coord[:2]  # Return 2D coordinates

        # Draw rectangles around detected eyes and apply the transformation
        for (x, y, w, h) in eyes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Transform the top-left corner of the rectangle
            transformed_coord = transform_coordinates(np.array([x, y]), transformation_matrix)
            print(f"Original: ({x}, {y}), Transformed: {transformed_coord}")

        # Display the output
        cv2.imshow('Eye Detection with Transformation', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("No image selected.")
