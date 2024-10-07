import cv2
import os
import shutil

def variance_of_laplacian(image):
    """ Compute the variance of the Laplacian, the result is a measure of the sharpness of the image. """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def filter_and_move_images(source_folder, dest_folder, threshold=200.0):
    """ Filter out blurred images below a certain threshold of sharpness and move sharp images to a new folder. """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # Create destination folder if it doesn't exist

    for filename in os.listdir(source_folder):
        if filename.lower().endswith('.jpg'):
            source_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(dest_folder, filename)
            image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                sharpness = variance_of_laplacian(image)
                if sharpness >= threshold:
                    shutil.move(source_path, dest_path)
                    print(f"Moved sharp image {filename} to {dest_folder} (sharpness = {sharpness:.2f})")
                else:
                    print(f"Image {filename} is blurred (sharpness = {sharpness:.2f})")

# Example usage
source_folder = 'playground/commonspace/input'
dest_folder = 'playground/commonspace/input/best'
filter_and_move_images(source_folder, dest_folder)

