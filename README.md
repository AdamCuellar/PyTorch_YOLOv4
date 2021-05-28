# YOLOv4

This is another PyTorch implementation of [YOLOv4](https://github.com/AlexeyAB/darknet). This is based on [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u3_preview).

* [[original Darknet implementation of YOLOv4]](https://github.com/AlexeyAB/darknet)

### development log

<details><summary> <b>Expand</b> </summary>

* `2021-05-28` - Clone [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u3_preview) u3_preview branch.

</details>

## Pretrained Models & Comparison

* Will add later

## Requirements

```
pip install -r requirements.txt
```
※ For running Mish models, please install https://github.com/thomasbrandon/mish-cuda

## Training

```
python train.py --device 0 --batch-size 16 --img 640 640 --data coco.yaml --cfg cfg/yolov4-pacsp.cfg --weights '' --name yolov4-pacsp
```

## Testing

```
python test.py --img 640 --conf 0.001 --batch 8 --device 0 --data coco.yaml --cfg cfg/yolov4-pacsp.cfg --weights weights/yolov4-pacsp.pt
```

## Citation

```
@article{bochkovskiy2020yolov4,
  title={{YOLOv4}: Optimal Speed and Accuracy of Object Detection},
  author={Bochkovskiy, Alexey and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2004.10934},
  year={2020}
}
```

```
@inproceedings{wang2020cspnet,
  title={{CSPNet}: A New Backbone That Can Enhance Learning Capability of {CNN}},
  author={Wang, Chien-Yao and Mark Liao, Hong-Yuan and Wu, Yueh-Hua and Chen, Ping-Yang and Hsieh, Jun-Wei and Yeh, I-Hau},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={390--391},
  year={2020}
}
```

## Acknowledgements

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
