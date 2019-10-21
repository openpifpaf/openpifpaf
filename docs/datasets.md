# Datasets

Download MSCOCO data:

```sh
mkdir data-mscoco
cd data-mscoco
gsutil ls gs://images.cocodataset.org  # to list available directories

mkdir -p images/val2017
gsutil -m rsync gs://images.cocodataset.org/val2017 images/val2017

mkdir -p images/train2017
gsutil -m rsync gs://images.cocodataset.org/train2017 images/train2017

gsutil cp gs://images.cocodataset.org/annotations/annotations_trainval2017.zip .
# or
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
unzip image_info_test2017.zip

# test images: run inside of images directory
wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip
```

Download MPII data:

```sh
mkdir data-mpii
cd data-mpii
wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip
```


Download NightOwls:

```sh
mkdir data-nightowls
cd data-nightowls
wget http://www.robots.ox.ac.uk/\~vgg/data/nightowls/python/nightowls_validation.json
wget http://www.robots.ox.ac.uk/\~vgg/data/nightowls/python/nightowls_validation.zip
unzip nightowls_validation.zip
```
