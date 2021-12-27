import json
import logging
import os
import re

import numpy as np
from tqdm import tqdm

from model import VARIANT_KEY, indexer

logger = logging.getLogger(__file__)


def cossim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))


def semantic_search(work_dir, query_str, topK=None):
    index_file_path = os.path.join(work_dir, '%s_index.json' % VARIANT_KEY)
    if not os.path.exists(index_file_path):
        logger.info("no index yet!")
        return []
    else:
        index = json.load(open(index_file_path))
        # {image_path:md5}
    
    imgfeats_root = os.path.join(work_dir, 'image_features', VARIANT_KEY)
    if not os.path.exists(imgfeats_root):
        logger.error("index present, but features directory not found!")
        return []    

    ## Features for Query string

    # remove junk
    regex = re.compile('[^a-zA-Z0-9]')
    clean_querystr = regex.sub(' ', query_str)
    logger.info("clean query - %s" % clean_querystr)

    query_feat = indexer.index_text(clean_querystr)
    logger.info("matching with %d gallery features" % len(index))

    imgpth_score = dict()
    for img_path, md5 in tqdm(index.items()):
        imgfeat_pth = os.path.join(imgfeats_root, '%s.npy' % md5)
        imgfeat = np.load(imgfeat_pth)

        score = cossim(imgfeat.flatten(), query_feat.flatten())
        imgpth_score[img_path] = score

    # TODO: set some threshold on the score to filter
    
    ranked_imgpths_it = sorted(imgpth_score.keys(), key=lambda x:imgpth_score[x], reverse=True)
    ranked_imgpths = list(ranked_imgpths_it[:min(topK, len(index))]) if topK is not None else list(ranked_imgpths_it)
    
    return ranked_imgpths