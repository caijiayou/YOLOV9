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

# 輸入資料集路徑
path = input("\n    請輸入dataset資料夾位置: ")
dataset_dir = str(path)
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# 新結構：images/train, images/test, images/val
train_images_dir = os.path.join(images_dir, 'train')
test_images_dir = os.path.join(images_dir, 'test')
val_images_dir = os.path.join(images_dir, 'val')
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)

# 新結構：labels/train, labels/test, labels/val
train_labels_dir = os.path.join(labels_dir, 'train')
test_labels_dir = os.path.join(labels_dir, 'test')
val_labels_dir = os.path.join(labels_dir, 'val')
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# 列出所有圖片
all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png')) and os.path.isfile(os.path.join(images_dir, f))]

# 隨機打亂
random.shuffle(all_images)

# 分割比例
split1 = int(0.9 * len(all_images))
split2 = int(0.95 * len(all_images))

train_images = all_images[:split1]
test_images = all_images[split1:split2]
val_images = all_images[split2:]

# 執行移動
move_files(images_dir, labels_dir, train_images_dir, train_labels_dir, train_images)
move_files(images_dir, labels_dir, test_images_dir, test_labels_dir, test_images)
move_files(images_dir, labels_dir, val_images_dir, val_labels_dir, val_images)

# 複製 classes.txt 到每個 labels 子資料夾中
classes_file = os.path.join(labels_dir, 'classes.txt')
if os.path.exists(classes_file):
    for subdir in [train_labels_dir, test_labels_dir, val_labels_dir]:
        shutil.copy(classes_file, subdir)
        print(f"已複製 classes.txt 到 {subdir}")
else:
    print("⚠️ 找不到 classes.txt，無法複製")