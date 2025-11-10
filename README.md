# STORM
This README.md is designed to help you run our model and to provide a deeper understanding of how to train it and evaluate the results.
## set up environment
First, please set up the environment required to run the SOM for obtaining the mask, or to run the TOM for detecting tracking loss.
```bash
conda create -n STORM python=3.10
conda activate STORM

# Installing requirements for SOM and TOM
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
# pip install ultralytics git+https://github.com/openai/CLIP.git pillow matplotlib
pip install sam2

# install environment for FoundationPose
cd FoundationPose/
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"
python -m pip install -r requirements.txt
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
pip install ninja   
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e . --no-deps
cd ..
pip install cmake
conda install boost boost-cpp
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
pip uninstall numpy scipy scikit-learn
pip install numpy==1.26.4
cd ..
```

## download the dataset
If you would like to evaluate our model on a BOP Challenge subdataset, please begin by downloading the dataset to your local hard drive.
```bash
mkdir dataset
cd dataset
# download the ycbv datasets, you can also download other datasets by change the DATASET_NAME
export DATASET_NAME=ycbv
huggingface-cli download bop-benchmark/$DATASET_NAME \
    --local-dir ./${DATASET_NAME}/ --repo-type=dataset
```

## training the som model
If you want to train the SOM model yourself, you can run the code below. It is very helpful for training the SOM on your own.
```bash
cd som
python pure_all.py --data_root "xxxxx"
python gemma_description.py --data_root "xxxx"
# use 8 gpu server to training the model
torchrun --nproc_per_node=8 scripts/train.py --config configs/train.yaml 
```

## test the som model
After obtaining the checkpoint—either by training it yourself or by using our pre-trained checkpoint—you can test the results for AP, IoU, and runtime, as well as generate the mask results using the code below.
```bash
# SOM without SAM2 for one picture
python SOM.py --query "xxxx" --views "xxxxx" --configs "xxxx"
# SOM with SAM2 for one picture
python SOM.py --query "xxxx" --views "xxxxx" --configs "xxxx"
# evaluate SOM in the datasets.
python evaluation_SOM_mask.py --data_root "xxxx"
```


## test the 6D Pose estimation result
After obtaining the result JSON file from the previous steps, you can calculate the evaluation metrics using this code.
```bash
python evaluation_result --bop_dir "xxxx" --pred_dir "xxxx"
```
## train the tom model
if you have the dataset of tracking, you can use it to train your tom model.
```bash
python ./tom/train/train.py 
```
## test the tom model on the tracking dataset
after training, you will have the checkpoint to run evaluation on the test dataset, you are supposed to get the accuracy and inference time.
```bash
python ./tom/evaluation/evaluationdino2.py --checkpoint "xxxx" --pairs-jsonl "xxxx" 
```

## 

## generate the 6D Pose estimation result from FP+SOM(GT)
You can generate the 6D pose estimation results using FoundationPose, with the masks provided either by SOM or from ground truth.
```bash
python evaluation_SOM.py --bop_dir "xxxx"
```

## generate the 6D Pose estimation tracking result from FP+STROM
You can test the 6D pose estimation tracking results from FoundationPose, either with TOM or on its own. The mask will be provided by SOM if re-registration is required.
```bash
python evaluation_STORM.py --bop_dir "xxxx"
```
