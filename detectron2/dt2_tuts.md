# Detectron 2 Tutorial#
Tutorial notes of detectron2, going over the basic usage, including:
- running inference on images or videos.
- train a detectron2 model on a new dataset.

#### Installing dependencies
1. Using PyTorch and torchvision; OpenCV is pre-installed on colab. (#Currently my local machine does not support runnning these frameworks, so using Colab as the preferred alternative)
```
# install dependencies: 
!pip install pyyaml==5.1 pycocotools>=2.0.1
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
!gcc --version
# opencv is pre-installed on colab
```
2. Installing detectron2 (colab has CUDA 1-.1 + torch 1.6)
```
# install detectron2: (colab has CUDA 10.1 + torch 1.6)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
assert torch.__version__.startswith("1.6")
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
```
3. Basic setup
- Setting up detectron2 logger
```
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
```
- Importing common libraries
```
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow
```
- Importing common detectron2 libraries
```
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
```
---

## Run a pre-trained detectron2 model

- We download an image from the COCO dataset.
![Downloaded-image](http://images.cocodataset.org/val2017/000000439715.jpg)
```python
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
im = cv2.imread("./input.jpg")
cv2_imshow(im)
```
-We create a detectron2 config and detectron2 `DefaultPredictor` to run inference on this image.
> Inference means, letting the NN apply its learned skills to new data.

```python
cfg = get_cfg()
#add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
#Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
```
- The above code needs to be run with a GPU, i guess. Using the Runtime on colab to run with no Hardware accelerator lead to problems.
```python
# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)
```
- The output can be seen by using a visualizer to draw predictions on the image.
```python
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(out.get_image()[:, :, ::-1])
```
---

## Training on custom dataset

To start off, the tutorial shows how to training an existing detectron2 model on a custom dataset in a new format.
- the balloon dataset is used. 
- we will train a balloon segmentation model from an existing model pre-trained on COCO dataset, which is available in detectron2's model zoon.

#### Preparing the dataset
- The COCO dataset does not have a balloon category and we should be able to recognize this new class in a few minutes.
```python
# download, decompress the data
!wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
!unzip balloon_dataset.zip > /dev/null
```
- Register the balloon dataset to detectron2, following the detectron2 custom dataset tutorial. Since, the dataset is in its custom format, a function is written to parse it into detectron2's standard format. __DO THE TUTORIAL__

```python
#if your dataset is in COCO format, this cell can be replaced by the following three lines:
#from detectron2.data.datasets import register_coco_instances
#register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
#register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")
```

- To verify the data loading is correct, lets visualize the annotations of randomly selected samples in the training set.
>WORKS

#### TRAIN
We fine tune a COCO-pretrained R50-FPN Mask R-CNN model on the balloon dataset. It takes 6 minutes to train 300 iterations.
```python
from detectron2.engine import DefaultTrainer
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  #Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    #300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
```

- The training curves can be investigated on the training board.
<br/>Look at training curves in tensorboard:
```python
%load_ext tensorboard
%tensorboard --logdir output
```
---
## Inference and Evaluation using the Trained model
We run an inference with the trained model on the balloon validation dataset. 
- Creating a predictor using the model we just trained:
```python
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   #set a custom testing threshold for this model
cfg.DATASETS.TEST = ("balloon_val", )
predictor = DefaultPredictor(cfg)
```
- Then, random samples are selected to visualize the prediction results.
```python
from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_balloon_dicts("balloon/val")
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])
```
- The performance can be evaluated using the AP metric implemented in COCO API.


```python

```
