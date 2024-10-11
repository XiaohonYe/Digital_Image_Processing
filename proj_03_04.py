from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.signal import medfilt2d

def load_image(path):
    image = Image.open(path).convert('L')  # Convert to grayscale
    return np.array(image)

def uniform_filtering(img, size):
    img_avg_filtered = uniform_filter(img, size=(size, size))
    return img_avg_filtered

def mediam_filtering(img, kernel_size):
    img_median_filtered = medfilt2d(img, kernel_size=kernel_size)
    return img_median_filtered

if __name__ == "__main__":
    img = load_image('./data/images_chapter_05/Fig5.10(a).jpg')

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')

    # 均值滤波
    img_avg_filtered = uniform_filtering(img, 3)
    axs[1].imshow(img_avg_filtered, cmap='gray')
    axs[1].set_title('Average Filter')
    axs[1].axis('off')

    # 中值滤波
    img_median_filtered = mediam_filtering(img, kernel_size=3)
    axs[2].imshow(img_median_filtered, cmap='gray')
    axs[2].set_title('Median Filter')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('proj_03_04.png', format='png')
