# Evaluation Logs

```
# python -m openpifpaf.eval_coco --checkpoint outputs/resnet101block5-pif-paf-edge401-190313-100107.pkl --long-edge=641 --loader-workers=8
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
