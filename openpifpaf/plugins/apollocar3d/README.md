# openpifpaf_apollocar3d

Car keypoints plugin for [OpenPifPaf](https://github.com/vita-epfl/openpifpaf).<br />
[__New__ 2021 paper](https://arxiv.org/abs/2103.02440):

> __OpenPifPaf: Composite Fields for Semantic Keypoint Detection and Spatio-Temporal Association__<br />
> _[Sven Kreiss](https://www.svenkreiss.com), [Lorenzo Bertoni](https://scholar.google.com/citations?user=f-4YHeMAAAAJ&hl=en), [Alexandre Alahi](https://scholar.google.com/citations?user=UIhXQ64AAAAJ&hl=en)_, 2021.
>

## Setup

```
pip3 install openpifpaf
```

(in case CUDA 9 as driver: 
` pip install torch==1.7.0+cu92 torchvision==0.8.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html`)

## Predict 
Prediction runs as standard openpifpaf predict command, but using the pretrained model on vehicles. The flag "--checkpoint shufflenetv2k16-apollo-24" will cause that our 24 kp version of the Shufflenet 16 (AP 76.1%) will be automatically downloaded. As an example, run the command:
```
python -m openpifpaf.predict \
<image path> \
--checkpoint shufflenetv2k16-apollo-24 -o \
--instance-threshold 0.07 --seed-threshold 0.07 \
--line-width 3 --font-size 0 --white-overlay 0.6 
```

## Preprocess Dataset
The preprocessing step converts the annotations into the standard COCO format. It creates a version with all 66 keypoints and also creates a sparsified version with 24 keypoints. The resulting pose can be obtained running:
`python -m openpifpaf_apollocar3d.utils.constants`
<!---
@Lorenzo: I am not showing this in the readme as the gif files are to big for version control.
More info here:  https://github.com/vita-epfl/openpifpaf/pull/376#issuecomment-812492917
//<img src="docs/CAR_24_Pose.gif" width="600"/>
//<img src="docs/CAR_66_Pose.gif" width="600"/>
-->

Create (or soft link) the following directories:
* `mkdir data outputs apollo-coco, apollo-coco/images apollo-coco/annotations`
* download and soft link to apollocar3d dataset into `data`
    
```
pip install pandas
pip install opencv-python==4.1.2.30
```
```
python -m openpifpaf_apollocar3d.apollo_to_coco
```

This script will create annotations with 24kps and 66kps simultaneously. The argument `--split_images` copies the original images in the new folders according to the train val split, slowing down the process. No need to use it multiple times.


## Train
The default is training with 66kps
Square-edge 769 (AP 76.1%)

```
python3 -m openpifpaf.train --dataset apollo \
--basenet=shufflenetv2k16 --apollo-square-edge=769 \
--lr=0.00002 --momentum=0.95  --b-scale=5.0 \
--epochs=300 --lr-decay 160 260 --lr-decay-epochs=10  --weight-decay=1e-5 \
--weight-decay=1e-5  --val-interval 10 --loader-workers 16 --apollo-upsample 2 \
--apollo-bmin 2 --batch-size 8
```

For smaller memory GPUs: square-edge 513

```
python3 -m openpifpaf.train --dataset apollo \
--basenet=shufflenetv2k16w --apollo-square-edge=513 \
--lr=0.00001 --momentum=0.98 --b-scale=20.0  --epochs=200 \
--lr-decay 130 140 --lr-decay-epochs=10  --weight-decay=1e-5  --loader-workers 16 \
  --val-interval 10 --batch-size 8 --apollo-upsample 2 --apollo-bmin 5
```

To train with 24kps you need to use the following command

```
python3 -m openpifpaf.train --dataset apollo \
--basenet=shufflenetv2k16 --apollo-square-edge=769 \
--lr=0.00002 --momentum=0.95  --b-scale=5.0 \
--epochs=300 --lr-decay 160 260 --lr-decay-epochs=10  --weight-decay=1e-5 \
--weight-decay=1e-5  --val-interval 10 --loader-workers 16 --apollo-upsample 2 \
--apollo-bmin 2 --batch-size 8 --apollo-use-24-kps --apollo-val-annotations \
<PathToThe/apollo_keypoints_24_train.json>
```


## Evaluation
With 66 kps, replace shufflenetv2k16-apollo-66 with a path to your own checkpoint, if you want to evaluate on your own model:
```
CUDA_VISIBLE_DEVICES=0,1 python3 -m openpifpaf.eval --dataset=apollo \
--checkpoint shufflenetv2k16-apollo-66 \
--force-complete-pose --seed-threshold=0.01 --instance-threshold=0.01 \
--apollo-eval-long-edge 0
```

With 24 kps, replace shufflenetv2k16-apollo-24 with a path to your own checkpoint, if you want to evaluate on your own model. Note that also in evaluation flag you need to make sure to set the cli flag for using 24kps only:
```
CUDA_VISIBLE_DEVICES=0,1 python3 -m openpifpaf.eval --dataset=apollo \
--checkpoint <PathToYourCheckpoint> \
--force-complete-pose --seed-threshold=0.01 --instance-threshold=0.01 \
--apollo-eval-long-edge 0 --apollo-use-24-kps --apollo-val-annotations \
<PathToThe/apollo_keypoints_24_train.json>
```

## Everything else
All pifpaf options and commands still stand, read more in the
[OpenPifPaf guide](https://vita-epfl.github.io/openpifpaf/intro.html)
