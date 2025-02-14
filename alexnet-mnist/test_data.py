import cv2
import numpy as np
import idx2numpy

def prepare_test_data(image_paths, labels_path, output_file):
    """
    準備測試數據並生成 C header 文件
    """
    header_content = "#include <stdint.h>\n"
    header_content += "#ifndef _TEST_DATA_H_\n#define _TEST_DATA_H_\n\n"
    header_content += f"#define NUM_TEST_IMAGES {3000}\n\n"
    
    labels = idx2numpy.convert_from_file(labels_path)
    labels = labels[:3000]
    images = idx2numpy.convert_from_file(image_paths)
    images = images[:3000]
    
    # 處理每張圖片
    for idx, (img, label) in enumerate(zip(images, labels)):
        img = cv2.resize(img, (28, 28))  # 調整到 28x28 大小
        img = (img.astype(np.float32) - 128).astype(np.int8)  # 將像素值轉為 -128 ~ 127
        
        # 轉換為 C 數組格式
        header_content += f"static const uint32_t test_image_{idx}[] = {{\n    "
        
        for i in range(0, 784, 4):
            # 提取 4 個像素值
            pixels = img.flatten()[i:i+4]
            while len(pixels) < 4:
                pixels = np.append(pixels, 0)  # 用 0 填充到 4 個像素
            
            # 處理每個像素，保證是 uint8 範圍內進行位運算
            packed_value = (int(pixels[0]) & 0xFF) | \
                           ((int(pixels[1]) & 0xFF) << 8) | \
                           ((int(pixels[2]) & 0xFF) << 16) | \
                           ((int(pixels[3]) & 0xFF) << 24)
            
            if i > 0 and i % 16 == 0:  # 每 16 個值換行
                header_content += "\n    "
            header_content += f"0x{packed_value:08X}, "
        
        header_content = header_content.rstrip(", ")
        header_content += "\n};\n\n"
    
    header_content += "const uint32_t* test_images[NUM_TEST_IMAGES] = {\n"
    for i in range(3000):
        header_content += f'\ttest_image_{i},\n'
    
    # 加入標籤數組
    header_content += "};\n\nstatic const uint8_t test_labels[] = {\n    "
    for idx, label in enumerate(labels):
        if idx > 0 and idx % 10 == 0:
            header_content += "\n    "
        header_content += f"{label}, "
    
    header_content = header_content.rstrip(", ")
    header_content += "\n};\n\n#endif // _TEST_DATA_H_"
    
    # 寫入檔案
    with open(output_file, "w") as f:
        f.write(header_content)

# 使用範例
if __name__ == "__main__":
    # 假設你有一個包含圖片路徑和標籤的列表
    image_paths = 'mnist_data/t10k-images.idx3-ubyte'
    labels = 'mnist_data/t10k-labels.idx1-ubyte'
    prepare_test_data(image_paths, labels, "test_data.h")
