conda create -n clip python=3.8
conda activate clip
conda install pytorch==1.7.1 torchvision==0.8.2 cpuonly -c pytorch
python -m pip install ftfy==6.1.1 regex==2022.6.2 tqdm==4.64.0 PySide6==6.3.1
python -m pip install git+https://github.com/openai/CLIP.git@b46f5ac7587d2e1862f8b7b1573179d80dcdd620