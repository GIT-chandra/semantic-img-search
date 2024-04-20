import json
import logging
import os

import numpy as np
from tqdm import tqdm

from model import get_md5sum, indexer, VARIANT_KEY

logger = logging.getLogger(__file__)


image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.jiff']


def index_images(images_root, work_dir):
    # compile images list
    all_images = []
    for (dirname, _, files_list) in os.walk(images_root):
        images_list = [os.path.abspath(os.path.join(dirname, x)) for x in filter(
            lambda ii: os.path.splitext(ii)[-1].lower() in image_extensions, files_list)]
        all_images.extend(images_list)
        logger.info("picked up %d images from %s" % (len(images_list), dirname))
    logger.info("%d images in total, under %s" % (len(all_images), images_root))

    # filter images which are not yet indexed
    index_file_path = os.path.join(work_dir, '%s_index.json' % VARIANT_KEY)
    if not os.path.exists(index_file_path):
        images_to_index = all_images
        index = dict()
        logger.info("no index yet!")
    else:
        index = json.load(open(index_file_path))
        images_to_index = list(filter(lambda x: x not in index, all_images))
        logger.info("currently %d images indexed" % len(index))
    logger.info("%d images need to be indexed" % len(images_to_index))

    imgfeats_root = os.path.join(work_dir, 'image_features', VARIANT_KEY)
    if not os.path.exists(imgfeats_root):
        os.makedirs(imgfeats_root)

    # start indexing
    for imgpth in tqdm(images_to_index, desc='Images indexing'):
        md5 = get_md5sum(open(imgpth, 'rb').read())
        imgfeat = indexer.index_image(imgpth)

        save_path = os.path.join(imgfeats_root, '%s.npy' % md5)
        np.save(save_path, imgfeat)

        index[imgpth] = md5

    # update index
    with open(index_file_path, 'w') as f:
        f.write(json.dumps(index))
    




    
   