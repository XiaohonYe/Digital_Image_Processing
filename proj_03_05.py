from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_image(path):
    image = Image.open(path).convert('L')  # Convert to grayscale
    return np.array(image)

def laplacian_filteringv1(img):
    # 默认的Laplacian滤波器（使用cv2.filter2D进行卷积操作）
    laplacian_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    img1 = cv2.filter2D(img, -1, kernel=laplacian_filter)  
    return img1

def laplacian_filteringv2(img):
    # 自定义Laplacian掩码
    laplacian_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    img2 = cv2.filter2D(img, -1,  laplacian_filter) 
    return img2

if __name__ == "__main__":
    img = load_image('./data/images_chapter_03/Fig3.40(a).jpg')

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Original')
    axs[0, 0].axis('off')

    img1 = laplacian_filteringv1(img)
    axs[1, 0].imshow(img1, cmap='gray')
    axs[1, 0].set_title('Default Laplacian')
    axs[1, 0].axis('off')

    # 自定义Laplacian掩码
    img2 = laplacian_filteringv2(img)
    axs[1, 1].imshow(img2, cmap='gray')
    axs[1, 1].set_title('Custom Laplacian Mask')
    axs[1, 1].axis('off')

    # 将原图与默认Laplacian滤波图相加
    img3 = cv2.add(img, img1)  # 使用cv2.add确保数据类型的一致性
    axs[2, 0].imshow(img3, cmap='gray')
    axs[2, 0].set_title('Output 1 (Original + Default Laplacian)')
    axs[2, 0].axis('off')

    # 将原图与自定义掩码滤波图相加
    img4 = cv2.add(img, img2)
    axs[2, 1].imshow(img4, cmap='gray')
    axs[2, 1].set_title('Output 2 (Original + Custom Laplacian Mask)')
    axs[2, 1].axis('off')

    # remove blank subplot
    fig.delaxes(axs[0, 1])  
    plt.tight_layout()
    plt.show()
    plt.savefig('hw_01/proj_03_05.png', format='png')
