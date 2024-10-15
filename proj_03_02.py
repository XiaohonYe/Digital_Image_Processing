from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['axes.unicode_minus'] = False

def load_image(image_path):
    image = Image.open(image_path).convert('L')
    return image

def histogram_equalization(image_array):
    # Flatten the image array and compute the histogram
    histogram, _ = np.histogram(image_array, bins=256, range=(0, 256))
    
    # # Compute the cumulative distribution function (CDF)
    cdf = histogram.cumsum()
    cdf_normalized = cdf * (255 / cdf[-1])  # Normalize to range 0-255

    # Use the CDF as a lookup table to transform the pixel values
    equalized_image_array = np.interp(image_array.flatten(), range(256), cdf_normalized).reshape(image_array.shape)
    
    # Convert back to unsigned 8-bit integer format
    equalized_image_array = np.round(equalized_image_array).astype(np.uint8)
    # equalized_image_array = exposure.equalize_hist(image_array) * 255 # anther way to histogram_equalization

    return equalized_image_array, cdf_normalized


def display_results(original_image, equalized_image):
    # Display original and equalized images
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')
    
    plt.subplot(2, 2, 3)
    plt.title("Equalized Image")
    plt.imshow(equalized_image, cmap='gray')
    
    plt.subplot(2, 2, 2)
    plt.title("Original Histogram")
    plt.hist(original_image.flatten(), bins=256, edgecolor = 'w')
    plt.xlim([0, 256])
    plt.ylim([0, 15000])
    
    plt.subplot(2, 2, 4)
    plt.title("Equalized Histogram")
    plt.hist(equalized_image.flatten(), bins=256, edgecolor='w')
    plt.xlim([0, 256])
    plt.ylim([0, 15000])

    plt.savefig('hw_01/proj_03_02.png', format='png')

if __name__ == "__main__":
    image_path = './data/images_chapter_03/Fig3.08(a).jpg'
    image = load_image(image_path)
    image_array = np.array(image)
    equalized_image_array, cdf_normalized = histogram_equalization(image_array)
    plt.figure()
    plt.plot(cdf_normalized)
    plt.title('直方图均衡化变换函数')
    plt.xlabel('原始像素值')
    plt.ylabel('均衡化后的像素值')
    plt.savefig('hw_01/proj_03_02_1.png', format='png')
    display_results(image_array, equalized_image_array)
