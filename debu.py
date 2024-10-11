import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为双精度浮点格式
I = cv2.imread('./data/images_chapter_03/Fig3.40(a).jpg', cv2.IMREAD_GRAYSCALE)
I = I.astype(np.float64) / 255.0  # 转换为[0,1]范围的双精度图像

# 初始化结果图像
J = np.zeros_like(I)

# 获取图像的尺寸
M, N = I.shape

# 滤波操作
for x in range(1, M-1):
    for y in range(1, N-1):
        J[x, y] = 9 * I[x, y]
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                J[x, y] -= I[x + dx, y + dy]

# 显示图像
plt.figure(figsize=(12, 4))

# 原始图像
plt.subplot(1, 3, 1)
plt.imshow((I * 255).astype(np.uint8), cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 滤波后图像
plt.subplot(1, 3, 2)
plt.imshow((J * 255).astype(np.uint8), cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

# 原图与滤波后图像相加
output = I + J
plt.subplot(1, 3, 3)
plt.imshow((output * 255).astype(np.uint8), cmap='gray')
plt.title('Original + Filtered')
plt.axis('off')

# 展示结果
plt.tight_layout()
plt.show()
plt.savefig('debug_0305.png')
