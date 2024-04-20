from PIL import Image
import numpy as np
import clip
import torch
import os
import re
import logging

from typing_extensions import Annotated
from fastapi import FastAPI, Body, UploadFile, File

def cossim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))
# from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn")
# logging.basicConfig(format='%(asctime)s %(levelname)s %(process)d %(filename)s:%(lineno)d %(message)s')
logger.setLevel("INFO")
logger.info("Testing LOGGGGGG")
logger.info("uvicorn has %d logger(s)" % len(logger.handlers))
logger.handlers[0].setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(process)d %(filename)s:%(lineno)d %(message)s'))
# logger.removeHandler()

# ch = logging.StreamHandler()
# ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(process)d %(filename)s:%(lineno)d %(message)s'))
# logger.addHandler(ch)
# logger.propagate = True

VARIANT = 'ViT-L/14@336px'
VARIANT_KEY = VARIANT.replace('/', '__')


class ImageTextIndexer(object):
    def __init__(self, model_type=VARIANT) -> None:
        super().__init__()
        logger.info('initializing indexer')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info('device - ' + str(self.device))
        logger.info('loading model...')
        self.model, self.preprocess = clip.load(model_type, device=self.device)
        logger.info('model loaded!')

    def index_image(self, imgfile):
        image = self.preprocess(Image.open(
            imgfile)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            return image_features.cpu().numpy()

    def index_text(self, query_text):
        text = clip.tokenize([query_text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            return text_features.cpu().numpy()


app = FastAPI()
indexer = ImageTextIndexer()

root_folder = os.path.join(os.path.dirname(__file__), 'api_data')

# ts_format = "%d %B %Y %I:%M:%S %p"


@app.get("/up_test/")
def api_hello():
    return {'message': 'Yaay! FastAPI it is!!!'}


@app.post("/index/")
def api_index(
        file_path: Annotated[str, Body(description='Path to file', embed=True)],
        file: Annotated[UploadFile, File(description='File to be indexed, an image')],
        db_name: Annotated[str, Body(description='db to use', embed=True)] = 'default'):

    logger.info(str(file.filename))
    logger.info('file mode - ' + str(file.file.mode))

    # TODO: validate this is actually an image file
    imgfeat = indexer.index_image(file.file)

    # TODO: file_path should not start with '/', neither db_name should have '/' - must be a valid folder name,
    # limit to simple chars
    save_path = os.path.join(root_folder, db_name, file_path)
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_path, imgfeat)

    return {'message': 'success'}


@app.post("/search/")
def api_search(
    query_string: Annotated[str, Body(description='Search term', embed=True)],
    search_path: Annotated[str, Body(
        description='Which folder to search in', embed=True)] = '',
    db_name: Annotated[str, Body(
        description='db to use', embed=True)] = 'default',
    recurse: Annotated[bool, Body(
        description='Whether to search in subfolders', embed=True)] = False,
):
    # TODO: 'recurse' implement

    regex = re.compile('[^a-zA-Z0-9]')
    clean_querystr = regex.sub(' ', query_string)
    logger.info("clean query - %s" % clean_querystr)

    query_feat = indexer.index_text(clean_querystr)

    search_folder = os.path.join(root_folder, db_name, search_path)
    if not os.path.exists(search_folder):
        logger.warning("folder not found! - %s" % search_folder)
        return {'message': 'success', 'results': []}

    imgpth_score = dict()
    for ff in os.listdir(search_folder):
        fpath = os.path.join(search_folder, ff)
        if os.path.isdir(fpath):
            continue
        imgfeat = np.load(fpath)

        score = cossim(imgfeat.flatten(), query_feat.flatten())
        imgpth_score[fpath] = score

    # TODO: set some threshold on the score to filter
    topK = 5

    ranked_imgpths_it = sorted(
        imgpth_score.keys(), key=lambda x: imgpth_score[x], reverse=True)
    ranked_imgpths = list(ranked_imgpths_it[:min(
        topK, len(imgpth_score))]) if topK is not None else list(ranked_imgpths_it)

    return {
        'message': 'success',
        'results': [x[len(os.path.join(root_folder, db_name)) + 1:-4] for
                    x in ranked_imgpths]
        # -4, for '.npy'
    }
