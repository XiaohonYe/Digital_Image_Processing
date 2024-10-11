from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    image = Image.open(path).convert('L')  # Convert to grayscale
    return np.array(image)

def save_image(image_array, path):
    image = Image.fromarray(image_array)
    image.save(path)

# Function to clip values to be within the range 0-255
def clip_values(image_array):
    return np.clip(image_array, 0, 255).astype(np.uint8)

# Arithmetic Operations

# 1. Addition of two images
def add_images(image1, image2):
    result = image1 + image2
    return clip_values(result)

# 2. Subtraction of two images
def subtract_images(image1, image2):
    result = image1 - image2
    return clip_values(result)

# 3. Multiplication of two images
def multiply_images(image1, image2):
    result = image1 * image2
    return clip_values(result)

# 4. Division of two images
def divide_images(image1, image2):
    # Avoid division by zero by replacing zeros in the denominator with a small value (epsilon)
    epsilon = 1e-8
    result = image1 / (image2 + epsilon)
    # Normalize the result to be within 0-255 if necessary
    result = (result * 255 / np.max(result))
    return clip_values(result)

# 5. Multiplication of an image by a constant
def multiply_image_by_constant(image, constant):
    result = image * constant
    return clip_values(result)

# Example Usage
if __name__ == "__main__":
    # Load two images
    image1 = load_image('path_to_image1.jpg')
    image2 = load_image('path_to_image2.jpg')

    # Perform operations
    added_image = add_images(image1, image2)
    subtracted_image = subtract_images(image1, image2)
    multiplied_image = multiply_images(image1, image2)
    divided_image = divide_images(image1, image2)
    constant_multiplied_image = multiply_image_by_constant(image1, 1.5)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(added_image, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title("Equalized Image")
    plt.imshow(subtracted_image, cmap='gray')
    plt.show()

    # Save the results
    # save_image(added_image, 'added_image.png')
    # save_image(subtracted_image, 'subtracted_image.png')
    # save_image(multiplied_image, 'multiplied_image.png')
    # save_image(divided_image, 'divided_image.png')
    # save_image(constant_multiplied_image, 'constant_multiplied_image.png')
