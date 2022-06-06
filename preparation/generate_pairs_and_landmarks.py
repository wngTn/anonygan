from train_images_and_keypoints_1 import *
import os
import shutil
import glob

images_path = '/home/tonyw/ba/anonygan/data/img_align_celeba'
landmark_file_name = 'test_landmarks.csv'
landmark_folder = os.path.join(images_path, 'testK_68')
picture_path = '/home/tonyw/ba/anonygan/pictures'


# the images for the swapping
images = glob.glob(picture_path + "/*.jpg")

# create all subdirs
for sub_dir in ['test', 'test_mask', 'testK_68', 'train']:
    os.makedirs(os.path.join(images_path, sub_dir), exist_ok=True)

# crop the images to H=220, W=180 and copy files to test and train folder for test.sh
for file in images:
    img = cv2.imread(file)
    if img.shape[:2] != (220, 180):
        H, W, _ = img.shape
        # resizing the width
        resized_W = int(H * (180/220))
        # if resized width bigger than original width, reduce height
        if resized_W > W:
            i = 1
            while int((H - i) * (180/220)) > W:
                i = i + 1
            new_H = H - i
            new_W = int(new_H * (180/220))
            # cropping
            crop_img = img[0:new_H, 0:new_W]
            # resizing
            img = cv2.resize(crop_img, (180, 220))
        else:
            # cropping
            crop_img = img[0:H, 0:resized_W]
            # resizing
            img = cv2.resize(crop_img, (180, 220))

    

    # writing image in test and train folder
    cv2.imwrite(os.path.join(images_path, os.path.basename(file)), img)
    cv2.imwrite(os.path.join(images_path, 'test', os.path.basename(file)), img)
    cv2.imwrite(os.path.join(images_path, 'train', os.path.basename(file)), img)

# path_generate_train_pairs(images_path)
detect_landmarks(images_path, landmark_file_name)
path_generate_landmarks(images_path, landmark_folder, landmark_file_name)