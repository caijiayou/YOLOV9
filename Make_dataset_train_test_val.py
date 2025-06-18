import os
import shutil
import random

def move_files(src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir, file_list):
    for file_name in file_list:
        base_name, _ = os.path.splitext(file_name)
        src_img = os.path.join(src_images_dir, file_name)
        dst_img = os.path.join(dst_images_dir, file_name)
        shutil.move(src_img, dst_img)
        
        label_file = base_name + '.txt'
        src_label = os.path.join(src_labels_dir, label_file)
        dst_label = os.path.join(dst_labels_dir, label_file)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        else:
            raise FileNotFoundError(f'Label file not found for {file_name}')

# 設定資料集目錄
path = input("\n    請輸入dataset資料夾位置: ")
dataset_dir = str(path)
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# 創建訓練、測試、驗證集目錄
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')
val_dir = os.path.join(dataset_dir, 'val')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 創建 train, test, val 的 images and labels 子目錄
train_images_dir = os.path.join(train_dir, 'images')
train_labels_dir = os.path.join(train_dir, 'labels')
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)

test_images_dir = os.path.join(test_dir, 'images')
test_labels_dir = os.path.join(test_dir, 'labels')
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

val_images_dir = os.path.join(val_dir, 'images')
val_labels_dir = os.path.join(val_dir, 'labels')
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# 列出所有圖片文件
images = os.listdir(images_dir)

# 隨機打亂資料
random.shuffle(images)

# 計算分割點
split1 = int(0.9 * len(images))
split2 = int(0.95 * len(images))

# 分割資料集
train_images = images[:split1]
test_images = images[split1:split2]
val_images = images[split2:]

# 移動圖片到對應目錄
move_files(images_dir, labels_dir, train_images_dir, train_labels_dir, train_images)
move_files(images_dir, labels_dir, test_images_dir, test_labels_dir, test_images)
move_files(images_dir, labels_dir, val_images_dir, val_labels_dir, val_images)