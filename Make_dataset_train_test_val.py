import os
import shutil
import random

def move_files(src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir, file_list):
    moved_count = 0  # 計算移動的圖片數量
    for file_name in file_list:
        base_name, _ = os.path.splitext(file_name)
        src_img = os.path.join(src_images_dir, file_name)
        dst_img = os.path.join(dst_images_dir, file_name)
        
        label_file = base_name + '.txt'
        src_label = os.path.join(src_labels_dir, label_file)
        dst_label = os.path.join(dst_labels_dir, label_file)

        if os.path.exists(src_label):
            shutil.move(src_img, dst_img)
            shutil.move(src_label, dst_label)
            moved_count += 1
        else:
            print(f'⚠️ 警告：找不到標籤檔，已跳過 {file_name}')
    return moved_count

# 輸入資料集路徑
path = input("\n    請輸入dataset資料夾位置: ")
dataset_dir = str(path).strip()
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# 建立新結構：images/train, images/test, images/val
train_images_dir = os.path.join(images_dir, 'train')
test_images_dir = os.path.join(images_dir, 'test')
val_images_dir = os.path.join(images_dir, 'val')
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)

# 建立新結構：labels/train, labels/test, labels/val
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
print("\n🚚 開始移動檔案...")
train_count = move_files(images_dir, labels_dir, train_images_dir, train_labels_dir, train_images)
test_count = move_files(images_dir, labels_dir, test_images_dir, test_labels_dir, test_images)
val_count = move_files(images_dir, labels_dir, val_images_dir, val_labels_dir, val_images)

print(f"\n✅ 資料集分割完成：")
print(f"訓練集：{train_count} 張圖片")
print(f"測試集：{test_count} 張圖片")
print(f"驗證集：{val_count} 張圖片")

# 複製 classes.txt 到每個 labels 子資料夾中
classes_file = os.path.join(labels_dir, 'classes.txt')
if os.path.exists(classes_file):
    for subdir in [train_labels_dir, test_labels_dir, val_labels_dir]:
        shutil.copy(classes_file, subdir)
        print(f"已複製 classes.txt 到 {subdir}")
else:
    print("⚠️ 找不到 classes.txt，無法複製")
