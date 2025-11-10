# SOM

## set up environment
conda create -n som python=3.10
conda activate som

pip install trimesh
pip install pyrender
pip install opencv-python
pip install jinja2 typeguard
pip install pyymal
pip install torch torchvision torchaudio
pip install tqdm
pip install transformers
pip install einops
pip install scikit-learn
pip install xformers --find-links https://github.com/facebookresearch/xformers/releases
pip install flash-attn --no-build-isolation
pip install albumentations
pip install pycocotools
pip install wandb

## download the dataset
mkdir dataset
cd dataset
export DATASET_NAME=ycbv
huggingface-cli download bop-benchmark/$DATASET_NAME \
    --local-dir ./${DATASET_NAME}/ --repo-type=dataset
    
## training the model
python process.py
python pure.py
python scripts/train.py --config configs/train.yaml
nohup python scripts/train.py --config configs/new_loss.yaml > AdaNorm_zero.log 2>&1 &
python scripts/train.py --config configs/new_loss.yaml
torchrun --nproc_per_node=8 --master_port=29501 scripts/train.py --config configs/train.yaml

# use multi GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 scripts/train.py --config configs/train.yaml 
torchrun --nproc_per_node=4 scripts/train.py --config configs/train.yaml  --resume ...
tmux new -s hsfa5 
torchrun --max-restarts=0 --nproc_per_node=8  --master_port=2952 scripts/train.py --config configs/train.yaml
## test the model
python scripts/infer.py 
  
  
