from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define dot patterns for levels 0 to 9
def halftone_pattern():
    dot_mat = np.zeros((10, 3, 3), dtype=np.uint8)
    dot_mat[1] = np.array([[0, 255, 0], [0, 0, 0], [0, 0, 0]])
    dot_mat[2] = np.array([[0, 255, 0], [0, 0, 0], [0, 0, 255]])
    dot_mat[3] = np.array([[255, 255, 0], [0, 0, 0], [0, 0, 255]])
    dot_mat[4] = np.array([[255, 255, 0], [0, 0, 0], [255, 0, 255]])
    dot_mat[5] = np.array([[255, 255, 255], [0, 0, 0], [255, 0, 255]])
    dot_mat[6] = np.array([[255, 255, 255], [0, 0, 255], [255, 0, 255]])
    dot_mat[7] = np.array([[255, 255, 255], [0, 0, 255], [255, 255, 255]])
    dot_mat[8] = np.array([[255, 255, 255], [255, 0, 255], [255, 255, 255]])
    dot_mat[9] = np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]])
    return dot_mat
    

def scale_image_if_needed(image, max_width_inches=8.5, max_height_inches=11, dpi=32):
    # Convert size from inches to pixels based on DPI (dots per inch)
    max_width_px = int(max_width_inches * dpi)
    max_height_px = int(max_height_inches * dpi)

    width, height = image.size

    # Calculate scaling factor
    width_scale = width / max_width_px
    height_scale = height / max_height_px
    scale = max(width_scale, height_scale)

    # If scaling needed
    if scale > 1:
        new_width = int(width / scale)
        new_height = int(height / scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def apply_halftoning(image):
    x_array = np.array(image.convert('L'))  
    qim = np.floor_divide(x_array, 25.6).astype(np.uint8)  # gray scale factor is 256 / 10 = 25.6

    ry, cy = qim.shape
    printed_img = np.zeros((ry * 3, cy * 3), dtype=np.uint8)

    dot_mat = halftone_pattern()

    # Apply halftoning
    for i in range(ry):
        for j in range(cy):
            level = qim[i, j]  # Get quantized level of each pixel
            printed_img[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = dot_mat[level] # each pixel in an input image will correspond to 3 x 3 pixels on the printed image

    return Image.fromarray(printed_img)

def main(image):
    original_image = image.convert("L")
    scaled_image = scale_image_if_needed(original_image, 8.5, 11, 32)
    
    halftoned_image = apply_halftoning(scaled_image)
    return halftoned_image


def create_test_pattern(size=256):
    test_image = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        test_image[:, i] = i
    return Image.fromarray(test_image)


if __name__ == "__main__":
    # Load and process an example image
    input_image_paths = ['./data/images_chapter_02/Fig2.22(a).jpg', 
                         './data/images_chapter_02/Fig2.22(b).jpg',
                         './data/images_chapter_02/Fig2.22(c).jpg']
    printed_images = []
    for iip in input_image_paths:
        input_image = Image.open(iip)
        # print(np.array(input_image).shape)
        halftoned_image = main(input_image)
        printed_images.append((input_image, halftoned_image))
    # print(np.array(halftoned_image).shape)

    # (b) Generate and save the test pattern
    test_pattern = create_test_pattern(256)
    halftoned_test_pattern = main(test_pattern)

    fig, axs = plt.subplots(2, 4, figsize=(20, 12))
    for col in range(3):
        image_pair = printed_images[col]
        axs[0, col].imshow(image_pair[0], cmap='gray')
        axs[0, col].set_title('Original')
        axs[0, col].axis('off')

        axs[1, col].imshow(image_pair[1], cmap='gray')
        axs[1, col].set_title('Halftoned Image')
        axs[1, col].axis('off')

    
    axs[0, 3].imshow(test_pattern, cmap='gray')
    axs[0, 3].set_title('Test Pattern Image')
    axs[0, 3].axis('off')

    axs[1, 3].imshow(halftoned_test_pattern, cmap='gray')
    axs[1, 3].set_title('Halftoned Test Pattern Image')
    axs[1, 3].axis('off')

    # remove blank subplot
    # fig.delaxes(axs[0, 1])  # 移除第一行第二个子图位置
    plt.tight_layout()
    plt.show()
    plt.savefig('hw_01/proj_02_01.png', format='png')

