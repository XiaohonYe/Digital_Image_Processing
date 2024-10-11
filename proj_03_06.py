from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_image(path):
    image = Image.open(path).convert('L')  # Convert to grayscale
    return np.array(image)

# 创建拉普拉斯滤波器
def create_laplacian_filterv1(alpha):
    return np.array([[0, -1, 0], [-1, 4 + alpha, -1], [0, -1, 0]])

def create_laplacian_filterv2(alpha):
    return np.array([[-1, -1, -1], [-1, 8 + alpha, -1], [-1, -1, -1]])



def main(img_path, mode):
    img = load_image(img_path)

    # 创建绘图窗口
    fig, axs = plt.subplots(2, 4, figsize=(10, 8))

    # 显示原始图像
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Original')
    axs[0, 0].axis('off')

    # 初始化计数器
    cnt = 0

    # 遍历不同的alpha值
    for alpha in [0.6, 1, 1.7, 2.3, 2.9, 5, 10]:
        # 创建Laplacian滤波器
        if mode == 'v1':
            h = create_laplacian_filterv1(alpha)
        else:
            h = create_laplacian_filterv2(alpha)

        # 应用卷积滤波器
        img_temp = cv2.filter2D(img, -1, h)

        # 将滤波后的图像与原图像相加
        img_out = cv2.add(img, img_temp)

        # 显示结果图像
        cnt += 1
        axs[cnt // 4, cnt % 4].imshow(img_out, cmap='gray')
        axs[cnt // 4, cnt % 4].set_title(f'α = {alpha}')
        axs[cnt // 4, cnt % 4].axis('off')
        
    # 调整布局以避免重叠
    plt.tight_layout()
    plt.show()
    plt.savefig(f'proj_03_06_{mode}.png', format='png')

if __name__ == "__main__":
    img_path = './data/images_chapter_03/Fig3.43(a).jpg'
    main(img_path, 'v2')
