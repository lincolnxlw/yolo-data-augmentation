from src.utils import get_inp_data, get_augmented_results, save_augmentation, is_image_by_extension
import argparse
from tqdm import tqdm
import yaml
import os

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)


def run_yolo_augmentor(need_save_bb_image=False,
                       is_test=False,
                       test_num=0):
    """
    Run the YOLO augmentor on a set of images.

    This function processes each image in the input directory, applies augmentations,
    and saves the augmented images and labels to the output directories.

    """
    #imgs = [img for img in os.listdir(CONSTANTS["inp_img_pth"]) if is_image_by_extension(img)]
    imgs = [img for img in os.listdir(config["inp_img_pth"])]

    for img_num, img_file in enumerate(tqdm(imgs, mininterval=1.0)):
        if is_test and img_num >= test_num:
            break

        print(f"Processing image {img_num+1}: {img_file}")
        ret, image, gt_bboxes, aug_file_name = get_inp_data(img_file=img_file, config=config)
        if not ret:
            continue
        aug_img, aug_label = get_augmented_results(image=image, bboxes=gt_bboxes)
        if len(aug_img) and len(aug_label):
            save_augmentation(trans_image=aug_img,
                              trans_bboxes=aug_label,
                              trans_file_name=aug_file_name,
                              need_save_bb_image=need_save_bb_image,
                              config=config)
        else:
            if len(aug_img) == 0:
                print(f"Augmented image is empty for {img_file}")
            if len(aug_label) == 0:
                print(f"Augmented label is empty for {img_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the YOLO augmentor on a set of images.")
    parser.add_argument('--need_save_bb_image', action='store_true', help='Save bounding box images.')
    parser.add_argument('--is_test', action='store_true', help='Test mode')
    parser.add_argument('--test_num', type=int, default=0, help='Number of images to test')
    args = parser.parse_args()

    need_save_bb_image = args.need_save_bb_image
    is_test = args.is_test
    test_num = args.test_num

    if not os.path.exists(config["out_img_pth"]):
        os.makedirs(config["out_img_pth"])
    if not os.path.exists(config["out_lab_pth"]):
        os.makedirs(config["out_lab_pth"])

    run_yolo_augmentor(need_save_bb_image=need_save_bb_image,
                       is_test=is_test,
                       test_num=test_num)