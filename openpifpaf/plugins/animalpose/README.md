# openpifpaf_apollocar3d


## Setup
```sh
pip install openpifpaf
pip install gdown
pip install scipy
pip install thop
```

### Dataset conversion
Download VOC dataset and the new images from the AnimalPose dataset and their annotations
```sh
mkdir data
mkdir data/animalpose
cd data/animalpose
pip install gdown
gdown https://drive.google.com/uc\?id\=1UkZB-kHg4Eijcb2yRWVN94LuMJtAfjEI
gdown https://drive.google.com/uc\?id\=1zjYczxVd2i8bl6QAqUlasS5LoZcQQu8b
gdown https://drive.google.com/uc\?id\=1MDgiGMHEUY0s6w3h9uP9Ovl7KGQEDKhJ
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar
tar -xvf keypoint_image_part2.tar.gz
tar -xvf keypoint_anno_part2.tar.gz
tar -xvf keypoint_anno_part1.tar.gz
tar -xvf VOCtrainval_25-May-2011.tar
rm keypoint_*.tar.gz
rm VOCtrainval_25-May-2011.tar
```



(in case CUDA 9 as driver: 
` pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html`)

* Download openpifpaf_animalpose, and create the following directories:
    * `mkdir data`
    * soft link output directory, which needs to be called outputs
    * soft link to animalpose dataset
    * create apollo-coco directory with `images/train`, `images/val`, `annotations` subdirectories and soft link them.

    
    
## Preprocess Dataset
`python -m openpifpaf_animalpose.voc_to_coco`
Use the argument `--split_images` to create a training val split copying original images in the new folders

## Show poses
`python -m openpifpaf_apollocar3d.utils.constants`

## Pretrained models
TODO

## Train
TODO

## Everything else
All pifpaf options and commands still hold, please check the 
[DEV guide](https://vita-epfl.github.io/openpifpaf/dev/intro.html)
