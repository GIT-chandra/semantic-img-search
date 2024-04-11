# semantic-img-search
Semantic Image Search


## Create Environment - CLIP
conda create -n semantis-clip python=3.8
conda activate semantis-clip
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
python -m pip install ftfy regex tqdm   # TODO: note down versions
python -m pip install git+https://github.com/openai/CLIP.git
