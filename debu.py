from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_image(path):
    image = Image.open(path).convert('L')  # Convert to grayscale
    return np.array(image)

def create_laplacian_filterv1(alpha):
    return np.array([[0, -1, 0], [-1, 4 + alpha, -1], [0, -1, 0]])

def create_laplacian_filterv2(alpha):
    return np.array([[-1, -1, -1], [-1, 8 + alpha, -1], [-1, -1, -1]])

def high_boost_filter(image, A):
    kernel = np.ones((3, 3), np.float32) / 9.0
    blurred_image = cv2.filter2D(image, -1, kernel)
    enhanced_image = (1 + A) * image - A * blurred_image
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    
    return enhanced_image, blurred_image

def main(img_path, mode):
    img = load_image(img_path)

    fig, axs = plt.subplots(2, 4, figsize=(10, 8))

    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Original')
    axs[0, 0].axis('off')

    cnt = 0

    for alpha in [0.6, 1, 1.7, 2.3, 2.9, 5]:
        
        enhanced_img, blurred_img = high_boost_filter(img, alpha)

        cnt += 1
        axs[cnt // 4, cnt % 4].imshow(enhanced_img, cmap='gray')
        axs[cnt // 4, cnt % 4].set_title(f'α = {alpha}')
        axs[cnt // 4, cnt % 4].axis('off')
        
    plt.tight_layout()
    plt.show()
    plt.savefig(f'hw_01/proj_03_06_{mode}.png', format='png')

if __name__ == "__main__":
    img_path = './data/images_chapter_03/Fig3.43(a).jpg'
    main(img_path, 'v2')
