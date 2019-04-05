# Evaluation Logs

## Low Resolution

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-190315-214317.pkl --long-edge=321 --loader-workers=8`:
```
```

## Low Resolution Ablation

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-l1-190403-162441.pkl --long-edge=321 --loader-workers=8 --fixed-b=1.0`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.419
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.688
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.427
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.266
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.627
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.715
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.483
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.698
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-smoothl1-rsmooth0.2-190401-114702.pkl --long-edge=321 --loader-workers=8 --fixed-b=1.0`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.691
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.426
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.270
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.719
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.480
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.697
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-smoothl1-rsmooth0.5-190403-162449.pkl --long-edge=321 --loader-workers=8 --fixed-b=1.0`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.689
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.422
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.271
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.627
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.717
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.697
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-smoothl1-rsmooth1.0-190401-114712.pkl --long-edge=321 --loader-workers=8 --fixed-b=1.0`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.418
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.687
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.426
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.266
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.715
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.483
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.699
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-190401-220105.pkl --long-edge=321 --loader-workers=8 --fixed-b=1.0`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.717
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.470
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.316
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.642
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.741
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.351
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.711
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-190401-220105.pkl --long-edge=321 --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.457
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.720
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.472
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.315
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.651
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.746
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.528
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.353
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.720
```


## High Resolution

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet50block5-pif-paf-edge401-190315-214317.pkl --long-edge=641 --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.631
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.857
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.692
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.603
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.678
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.686
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.884
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.741
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.639
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.751
Decoder 0: decoder time = 798.1136615276337s
total processing time = 868.1674637794495s
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet101block5-pif-paf-edge401-190313-100107.pkl --long-edge=641 --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.662
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.872
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.724
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.623
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.721
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.712
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.895
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.768
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.660
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.785
Decoder 0: decoder time = 776.0574517250061s
total processing time = 1015.5730197429657s
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet101block5-pif-paf-edge401-190313-100107.pkl --long-edge=641 --loader-workers=8 --pif-fixed-scale=0.5`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.650
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.861
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.712
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.618
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.705
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.705
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.887
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.760
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.657
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.773
```

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-edge401-190322-092459.pkl --long-edge=641 --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.679
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.875
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.742
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.636
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.743
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.726
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.898
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.781
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.800
```

Experiment with updating batch norm parameters: `python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-edge401-190328-112737.pkl --long-edge=641 --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.681
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.875
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.748
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.640
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.743
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.727
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.899
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.784
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.674
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.801
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

Experiment with updating batch norm parameters: `python -m openpifpaf.eval_coco --checkpoint outputs/resnet152block5-pif-paf-edge401-190328-112737.pkl --long-edge=641 --loader-workers=8 --dataset=test-dev --write-predictions --all-images`:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.667
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.881
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.736
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.626
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.727
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.723
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.911
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.784
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.666
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.799
```


# Timing on 1080Ti

`python -m openpifpaf.eval_coco --checkpoint outputs/resnet101block5-pif-paf-edge401-190313-100107.pkl --long-edge=641  --all-images --loader-workers=8`:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.658
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.866
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.719
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.619
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.718
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.712
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.895
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.768
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.660
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.785
Decoder 0: decoder time = 866.8524780273438s
total processing time = 1188.9627029895782s
```
