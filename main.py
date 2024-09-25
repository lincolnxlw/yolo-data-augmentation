from utils import *
import argparse
from tqdm import tqdm


def run_yolo_augmentor(need_save_bb_image=False):
    """
    Run the YOLO augmentor on a set of images.

    This function processes each image in the input directory, applies augmentations,
    and saves the augmented images and labels to the output directories.

    """
    imgs = [img for img in os.listdir(CONSTANTS["inp_img_pth"]) if is_image_by_extension(img)]

    for img_num, img_file in enumerate(tqdm(imgs, mininterval=1.0)):
        print(f"Processing image {img_num+1}: {img_file}")
        image, gt_bboxes, aug_file_name = get_inp_data(img_file)
        aug_img, aug_label = get_augmented_results(image, gt_bboxes)
        if len(aug_img) and len(aug_label):
            save_augmentation(aug_img, aug_label, aug_file_name, need_save_bb_image)
        else:
            if len(aug_img) == 0:
                print(f"Augmented image is empty for {img_file}")
            if len(aug_label) == 0:
                print(f"Augmented label is empty for {img_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the YOLO augmentor on a set of images.")
    parser.add_argument('--need_save_bb_image', action='store_true', help='Save bounding box images.')
    args = parser.parse_args()
    need_save_bb_image = args.need_save_bb_image

    run_yolo_augmentor(need_save_bb_image=need_save_bb_image)