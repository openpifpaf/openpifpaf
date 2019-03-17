# Evaluation Logs

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
