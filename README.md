# Real-time multi-object tracking using Yolov8 with OCSORT

<div align="center">
  <p>
  <img src="video/cars1_result.gif" width="500"/>
  </p>
</div>


## Installation

```bash
$ pip install -r requirements.txt
```

## Tracking

```bash
# YOLOv4
$ python track.py --yolov4 yolov4/model_data/yolov4_weights_coco.pth
# YOLOv4-tiny
$ python track.py --yolov4-tiny yolov4_tiny/model_data/yolov4_tiny_weights_coco.pth
```
  
- Tracking sources

  ```bash
  $ python track.py --source 0                               # webcam
                             img.jpg                         # image
                             vid.mp4                         # video
                             path/                           # directory
                             path/*.jpg                      # glob
                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
  ```

- Tracking methods

  ```bash
  $ python track.py --tracking-method ocsort
                                      deepocsort
                                      strongsort
                                      bytetrack
                                      botsort
  ```

- Filter tracked classes

  By default the tracker tracks all MS COCO classes.
  If you want to track a subset of the classes that you model predicts, add their corresponding index after the classes flag.

  ```bash
  $ python track.py --classes 0 2  # Track person and car, only
  ```

  [Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a Yolov8 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero