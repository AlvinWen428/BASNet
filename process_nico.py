import os
import argparse
import numpy as np
from PIL import Image

from get_nico_saliency import show


def crop(image, saliency):
    """
    image: np.array [w, h, 3]
    saliency: np.array [w, h]
    """
    # find the smallest bbox
    leftmost, rightmost, upmost, lowmost = 0, 0, 0, 0
    for i in range(saliency.shape[0]):
        if np.any(saliency[i, ...] > 0):
            upmost = i
            break
    for i in range(saliency.shape[0]-1, -1, -1):
        if np.any(saliency[i, ...] > 0):
            lowmost = i
            break
    for i in range(saliency.shape[1]):
        if np.any(saliency[:, i, ...] > 0):
            leftmost = i
            break
    for i in range(saliency.shape[1]-1, -1, -1):
        if np.any(saliency[:, i, ...] > 0):
            rightmost = i
            break

    output = image[upmost: lowmost, leftmost: rightmost]
    return output


def mask(image, saliency):
    """
        image: np.array [w, h, 3]
        saliency: np.array [w, h]
        """
    output = image * saliency[:, :, None]
    return output


def main(process_method):
    root_folder = '/data/cwen/NICO/'
    image_folder = '/data/cwen/NICO/multi_classification/'
    output_folder = '/data/cwen/NICO/{}_by_saliency'.format(process_method)

    process_func = globals()[process_method]

    data_partition_list = os.listdir(image_folder)
    for data_partition in data_partition_list:  # train, val, test
        image_name_list = os.listdir(os.path.join(image_folder, data_partition))
        for name in image_name_list:
            img_name = name.split('/')[-1].split('.')[0]
            saliency_mask_name = img_name + '.png'

            raw_image = np.array(Image.open(os.path.join(image_folder, data_partition, name)).convert('RGB'))
            saliency_mask = np.array(Image.open(os.path.join(root_folder, 'saliency_mask', data_partition, saliency_mask_name)))
            saliency_mask = saliency_mask / 255

            output_image = process_func(raw_image, saliency_mask)

            output_image_file = Image.fromarray(np.uint8(output_image))
            os.makedirs(os.path.join(output_folder, data_partition), exist_ok=True)
            output_image_file.save(os.path.join(output_folder, data_partition, img_name + '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='mask', choices=['crop', 'mask'])
    args = parser.parse_args()
    main(args.method)

