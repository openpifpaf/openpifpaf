# Evaluation Logs

## Low Resolution

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-190315-214317.pkl --long-edge=321 --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.500
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.735
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.529
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.359
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.697
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.550
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.760
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.764
```

## Low Resolution Ablation

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-l1-190403-162441.pkl --long-edge=321 --loader-workers=8 --fixed-b=1.0`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.685
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.426
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.265
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.715
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.483
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.698
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-smoothl1-rsmooth0.2-190401-114702.pkl --long-edge=321 --loader-workers=8 --fixed-b=1.0`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.689
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.424
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.269
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.719
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.480
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.697
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-smoothl1-rsmooth0.5-190403-162449.pkl --long-edge=321 --loader-workers=8 --fixed-b=1.0`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.419
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.686
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.270
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.717
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.697
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-smoothl1-rsmooth1.0-190401-114712.pkl --long-edge=321 --loader-workers=8 --fixed-b=1.0`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.416
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.684
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.424
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.265
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.623
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.715
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.483
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.699
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-190401-220105.pkl --long-edge=321 --loader-workers=8 --fixed-b=1.0`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.451
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.713
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.468
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.314
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.741
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.351
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.711
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-190401-220105.pkl --long-edge=321 --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.455
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.717
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.471
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.314
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.649
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.746
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.528
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.353
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.720
```


## High Resolution

WITH TIMING `python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-190315-214317.pkl --long-edge=641 --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.626
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.851
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.687
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.599
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.674
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.686
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.884
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.741
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.639
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.751
Decoder 0: decoder time = 888.3387265205383s
total processing time = 1107.5033490657806s
```

WITH TIMING `python -m openpifpaf.eval_coco --checkpoint outputs/resnet101block5-pif-paf-edge401-190313-100107.pkl --long-edge=641 --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.657
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.866
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.719
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.619
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.718
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.712
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.895
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.768
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.660
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.785
Decoder 0: decoder time = 875.4406125545502s
total processing time = 1198.353811264038s
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet101block5-pif-paf-edge401-190313-100107.pkl --long-edge=641 --loader-workers=8 --pif-fixed-scale=0.5`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.645
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.854
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.707
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.613
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.702
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.705
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.887
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.760
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.657
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.773
```

WITH TIMING `python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-edge401-190322-092459.pkl --long-edge=641 --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.674
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.869
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.738
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.631
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.741
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.726
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.898
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.781
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.800
Decoder 0: decoder time = 861.8545560836792s
total processing time = 1315.8536067008972s
```


### test-dev

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-edge401-190322-092459.pkl --long-edge=641 --loader-workers=8 --dataset=test-dev --write-predictions --all-images`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.667
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.878
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.736
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.624
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.729
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.722
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.909
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.783
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.664
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.800
```


# v0.5.0

## High Resolution

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-190424-122009.pkl --long-edge=641 --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.638
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.858
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.700
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.612
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.679
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.694
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.889
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.750
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.649
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.757
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet101block5-pif-paf-edge401-190412-151013.pkl --long-edge=641 --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.666
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.871
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.730
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.629
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.723
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.714
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.896
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.771
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.663
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.787
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-edge401-190412-121848.pkl --long-edge=641 --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.682
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.877
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.746
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.643
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.740
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.728
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.900
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.783
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.678
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.798
```

### test-dev

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-edge401-190412-121848.pkl --long-edge=641 --loader-workers=8 --dataset=test-dev --write-predictions --all-images`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.670
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.883
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.739
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.632
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.726
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.725
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.911
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.785
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.671
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.798
```

# 0.7.0

`time CUDA_VISIBLE_DEVICES=1 python -m openpifpaf.eval_coco --checkpoint resnet50 --long-edge=641 --all-images --loader-workers=8`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.633
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.848
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.693
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.603
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.681
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.694
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.886
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.749
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.647
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.760
Decoder 0: decoder time = 168.98903698846698s
total processing time = 374.1286737918854s
```

`time CUDA_VISIBLE_DEVICES=1 python -m openpifpaf.eval_coco --checkpoint resnet101 --long-edge=641 --all-images --loader-workers=8`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.660
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.859
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.721
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.621
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.719
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.714
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.892
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.768
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.664
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.785
Decoder 0: decoder time = 158.98870043922216s
total processing time = 486.5417158603668s
```

`time CUDA_VISIBLE_DEVICES=1 python -m openpifpaf.eval_coco --checkpoint resnet152 --long-edge=641 --all-images --loader-workers=8`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.676
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.865
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.740
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.637
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.735
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.728
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.896
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.784
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.678
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.798
Decoder 0: decoder time = 137.40693471115083s
total processing time = 595.8605711460114s
```

# 0.8.0

`time CUDA_VISIBLE_DEVICES=1 python -m openpifpaf.eval_coco --checkpoint resnet50 --long-edge=641 --all-images --loader-workers=8`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.633
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.850
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.695
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.603
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.682
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.692
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.885
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.747
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.758
Decoder 0: decoder time = 178.62342172674835s
total processing time = 384.1026601791382s
```

`time CUDA_VISIBLE_DEVICES=1 python -m openpifpaf.eval_coco --checkpoint resnet101 --long-edge=641 --all-images --loader-workers=8`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.664
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.861
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.729
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.626
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.722
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.718
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.896
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.775
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.666
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.790
Decoder 0: decoder time = 167.94371821917593s
total processing time = 495.8108365535736s
```

`time CUDA_VISIBLE_DEVICES=1 python -m openpifpaf.eval_coco --checkpoint resnet152 --long-edge=641 --all-images --loader-workers=8`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.677
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.868
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.744
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.641
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.735
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.729
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.900
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.788
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.680
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.798
Decoder 0: decoder time = 159.6303431801498s
total processing time = 620.2563769817352s
```

# v0.10.0

`time CUDA_VISIBLE_DEVICES=0,1 python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-paf25-edge401-190926-205058-3355106f.pkl --connection-method=blend --long-edge=641 --loader-workers=16 --batch-size=16 --decoder-workers=16`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.696
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.874
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.759
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.646
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.770
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.741
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.899
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.797
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.684
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.821
```

`time CUDA_VISIBLE_DEVICES=0,1 python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-paf25-edge401-190926-205058-3355106f.pkl --connection-method=blend --long-edge=961 --loader-workers=16 --batch-size=16 --decoder-workers=16 --multi-scale`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.728
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.887
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.792
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.689
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.788
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.777
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.912
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.831
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.725
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.850
```

paf-th=0.03: `time CUDA_VISIBLE_DEVICES=2,3 python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-paf25-edge401-191003-212408.pkl.epoch070 --connection-method=blend --long-edge=961 --loader-workers=16 --batch-size=16 --paf-th=0.03 --multi-scale`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.730
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.889
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.796
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.691
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.789
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.779
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.916
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.837
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.729
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.850
```

`time CUDA_VISIBLE_DEVICES=1 python -m openpifpaf.benchmark --backbones outputs/resnet152block5-pif-paf-paf25-edge401-191003-212408.pkl.epoch070 --paf-th=0.03`:

| Backbone           | AP       | APᴹ      | APᴸ      | t_{total} [ms]  | t_{dec} [ms] |
|-------------------:|:--------:|:--------:|:--------:|:---------------:|:------------:|
| singlescale-max    | __69.7__ | 65.1     | 76.6     | 151             | 55           |
| singlescale        | __69.8__ | 65.2     | 76.8     | 153             | 56           |
| multiscale-nohflip | __72.2__ | 68.2     | 78.4     | 853             | 317          |
| multiscale         | __73.0__ | 69.1     | 78.9     | 1507            | 433          |


### test-dev

`time CUDA_VISIBLE_DEVICES=2 python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-paf25-edge401-190926-205058-3355106f.pkl --connection-method=blend --long-edge=641 --loader-workers=8 --dataset=test-dev --write-predictions --all-images`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.682
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.883
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.751
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.635
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.748
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.737
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.914
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.798
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.679
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.815
```

`time CUDA_VISIBLE_DEVICES=2 python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-paf25-edge401-190926-205058-3355106f.pkl --connection-method=blend --long-edge=961 --loader-workers=8 --dataset=test-dev --write-predictions --all-images --multi-scale`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.711
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.892
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.784
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.673
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.768
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.771
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.925
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.832
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.718
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.843
```

`time CUDA_VISIBLE_DEVICES=2,3 python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-paf25-edge401-191003-212408.pkl.epoch070 --connection-method=blend --long-edge=961 --loader-workers=16 --batch-size=16 --paf-th=0.03 --multi-scale --dataset=test --all-images --write-predictions`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.712
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.894
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.785
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.674
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.769
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.773
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.928
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.833
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.720
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.845
```


### test

`time CUDA_VISIBLE_DEVICES=2,3 python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-paf25-edge401-190926-205058-3355106f.pkl --connection-method=blend --long-edge=961 --loader-workers=8 --dataset=test --write-predictions --all-images --multi-scale --batch-size=8 --decoder-workers=8`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.694
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.874
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.757
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.646
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.764
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.760
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.915
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.813
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.704
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.836
```

`time CUDA_VISIBLE_DEVICES=2,3 python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-paf25-edge401-191003-212408.pkl.epoch070 --connection-method=blend --long-edge=961 --loader-workers=16 --batch-size=16 --paf-th=0.03 --multi-scale --dataset=test --all-images --write-predictions`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.694
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.876
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.758
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.647
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.765
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.760
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.917
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.815
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.704
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.836
```
