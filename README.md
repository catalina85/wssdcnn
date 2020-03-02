# wssdcnn
TensorFlow implementation of [Weakly- and Self-Supervised Learning for Content-Aware Deep Image Retargeting](https://arxiv.org/abs/1708.02731).

## Requirements

- Python 3.0
- TensorFlow >= 1.12

## Pre-trained model for test
- Download the pre-trained model weights from [here(code:e9y4)](https://pan.baidu.com/s/1zZA1ZKz-5jZue6Gm8sKVig). 
- Move the param files to directory `model_ckpt`.

- `python test.py`.

## Results

Some good results(aspect_ratio = 0.5).

|original                       | wssdcnn                           | bilinear                        |
| --------------------------    | ------------------------------    | ------------------------------  |
| ![](output_dir/twobirds.png)  | ![](output_dir/twobirds_0.5.png)  | ![](output_dir/twobirds_bi.png) |
| ![](output_dir/fishing.png)   | ![](output_dir/fishing_0.5.png)   | ![](output_dir/fishing_bi.png)  |
| ![](output_dir/butterfly.png) | ![](output_dir/butterfly_0.5.png) | ![](output_dir/butterfly_bi.png)|
| ![](output_dir/eagle.png)     | ![](output_dir/eagle_0.5.png)     | ![](output_dir/eagle_bi.png)    |
| ![](output_dir/surfer.png)    | ![](output_dir/surfer_0.5.png)    | ![](output_dir/surfer_bi.png)   |
| ![](output_dir/bike.png)      | ![](output_dir/bike_0.5.png)      | ![](output_dir/bike_bi.png)     |
