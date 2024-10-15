from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['axes.unicode_minus'] = False

def load_image(path):
    image = Image.open(path).convert('L')  # Convert to grayscale
    return np.array(image)

def high_boost_filter(image, A):
    kernel = np.ones((3, 3), np.float32) / 9.0
    blurred_image = cv2.filter2D(image, -1, kernel)
    enhanced_image = A * image - blurred_image
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    
    return enhanced_image, blurred_image

def main(img_path):
    img = load_image(img_path)
    blurred_img = None
    fig, axs = plt.subplots(2, 4, figsize=(10, 6))

    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Original')
    axs[0, 0].axis('off')

    cnt = 0

    for alpha in [0.6, 1, 1.7, 2.3, 2.9, 5]:
        
        enhanced_img, blurred_img = high_boost_filter(img, alpha)
        cnt += 1
        axs[cnt // 4, cnt % 4].imshow(enhanced_img, cmap='gray')
        axs[cnt // 4, cnt % 4].set_title(f'A = {alpha}')
        axs[cnt // 4, cnt % 4].axis('off')
        
    axs[-1, -1].imshow(blurred_img, cmap='gray')
    axs[-1, -1].set_title('应用均值滤波的模糊图像')
    axs[-1, -1].axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'hw_01/proj_03_06.png', format='png')

if __name__ == "__main__":
    img_path = './data/images_chapter_03/Fig3.43(a).jpg'
    main(img_path)
