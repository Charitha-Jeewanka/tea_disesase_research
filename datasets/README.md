# Tea-Leaf Disease and Pest Detection Dataset (v1.0)

Multi-class object-detection dataset of tea-leaf pathologies and pests, prepared in
the YOLO format and used to train and evaluate **CA-YOLOv8n**, a lightweight,
Spatial-Aware variant of YOLOv8n described in:

> [Lyu X, Song C, Yu Y. *Spatial-Aware Lightweight Network for Real-time Tea
> Disease Detection: A Coordinate Attention-Enhanced YOLOv8n Approach with
> Path-Decoupling Strategy*. PLOS ONE, in revision.]

## Contents

| Split | Images | Labels |
|-------|--------|--------|
| train | 7,260  | 7,260  |
| valid | 784    | 784    |
| test  | 1,547  | 1,547  |
| **Total** | **9,591** | **9,591** |

Total size: ~366 MB (uncompressed).

## Directory layout

```
tea_disease_dataset/
├── data.yaml                # YOLOv8 dataset descriptor (class names + split paths)
├── README.md                # this file
├── train/
│   ├── images/              # 7,260 JPG images
│   └── labels/              # 7,260 YOLO-format .txt label files
├── valid/
│   ├── images/              # 784 JPG images
│   └── labels/              # 784 .txt files
└── test/
    ├── images/              # 1,547 JPG images
    └── labels/              # 1,547 .txt files
```

## Classes (8)

| ID | Name |
|----|------|
| 0  | Black rot of tea |
| 1  | Brown blight of tea |
| 2  | Leaf rust of tea |
| 3  | Red Spider infested tea leaf |
| 4  | Tea Mosquito bug infested leaf |
| 5  | Tea leaf |
| 6  | White spot of tea |
| 7  | disease |

## Label format

Each `.txt` label file follows the standard YOLO format. One line per object:

```
<class_id> <cx> <cy> <w> <h>
```

All bounding-box coordinates are normalised to `[0, 1]` relative to the image
dimensions. `cx`, `cy` denote box centre; `w`, `h` denote box width and height.

## How to use

The bundle is drop-in compatible with the [Ultralytics YOLO](https://docs.ultralytics.com/)
training pipeline. After cloning Ultralytics:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="path/to/tea_disease_dataset/data.yaml",
            epochs=150, imgsz=640, batch=16)
```

## Provenance and license

The underlying images were originally compiled from publicly accessible
tea-disease pages on the Tencent Developer Community
(<https://cloud.tencent.com/developer>) and subsequently re-organised through the
Roboflow data-preparation pipeline (the `.rf.<hash>` suffix on label filenames is
a Roboflow signature). The original distributor did not attach explicit licence
terms to the upload.

This bundle is re-hosted on figshare **solely to support reproducibility of the
findings reported in the manuscript cited above**. Users who wish to redistribute
or otherwise reuse the data beyond reproduction of those findings should consult
the original Tencent Developer Community source.

## Citation

If you use this dataset, please cite the manuscript above and link back to the
figshare record:

```
@article{lyu_caplusyolov8n,
  title   = {Spatial-Aware Lightweight Network for Real-time Tea Disease Detection:
             A Coordinate Attention-Enhanced YOLOv8n Approach with Path-Decoupling Strategy},
  author  = {Lyu, Xiang and Song, ChengLei and Yu, Yue},
  journal = {PLOS ONE},
  year    = {2026},
  note    = {Dataset: https://doi.org/10.XXXX/XXXXX}
}
```

## Contact

ChengLei Song · School of Management, Xinyang Agriculture and Forestry University
· `2015220006@xyafu.edu.cn`
