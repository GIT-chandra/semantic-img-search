
import hashlib

import numpy as np
import pandas as pd
from argparse import ArgumentParser

import os
import sys
import re
import logging

logger = logging.getLogger(__file__)

from indexer import index_images
# from .matcher import semantic_search


# def cossim(a,b):
#     return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))


def parse_program_args():
    parser = ArgumentParser(description='CLI tool for managing images')
    parser.add_argument('gallery_dir', type=str, help='Root folder of images')
    parser.add_argument('--query', type=str, help='Query text', default='')
    parser.add_argument('--batch_size', type=int, help='Batch size for search, default - 128', default=128)
    parser.add_argument('--action', type=str, help='Action to be performed - [search, index]. default - search', default='search')
    parser.add_argument('--work_dir', type=str, help='Folder to keep generated files. defaul - ./.sis_filest', default='./.sis_files')
    parser.add_argument('--out_dir', type=str, help='Folder to keep search results. default - ./sis_results', default='./sis_results')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(process)d - %(filename)s:%(lineno)d - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout, 
        level=logging.INFO
        )

    args = parse_program_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    if args.action == 'index':
        index_images(args.gallery_dir, args.work_dir)
    # elif args.action == 'search':
    #     if args.query:
    #         semantic_search(args.gallery_dir, args.work_dir, args.query, args.batch_size)
    #     else:
    #         logger.info("Please provide a search string (--query) to do a search")

   