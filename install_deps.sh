conda create -n clip python=3.8
conda activate clip
conda install pytorch==1.7.1 torchvision==0.8.2 cpuonly -c pytorch
python -m pip install ftfy regex tqdm
python -m pip install git+https://github.com/openai/CLIP.git