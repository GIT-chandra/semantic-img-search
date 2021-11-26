import hashlib

import torch
import clip
from PIL import Image

VARIANT = "ViT-B/32"
VARIANT_KEY = VARIANT.replace('/', '__')


def get_md5sum(contents):
    digestor = hashlib.md5()
    digestor.update(contents)
    return digestor.hexdigest()


class ImageTextIndexer(object):
    def __init__(self, model_type=VARIANT) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_type, device=self.device)

    def index_image(self, imgpath):
        image = self.preprocess(Image.open(imgpath)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            return image_features.cpu().numpy()
    
    def index_text(self, query_text):
        with torch.no_grad():
            text_features = self.model.encode_text(query_text)            
            return text_features.cpu().numpy()

indexer = ImageTextIndexer()