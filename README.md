## TODO

introduce model


# KDDF

## How to get the code
```
git clone https://github.com/melamaze/KDDF-pytorch-grad-cam
```

## How to dowload testing dataset

You can get testing dataset in this [link](https://drive.google.com/drive/u/0/folders/1r_P8PpbLATzMYk76sprbtSmpq972N5Xe).

Decompress dataset by:
```
unzip V1.zip  
unzip V2.zip
```

## How to excute KDDF
```python
# example: excute resnet18, V1, layercam 
python3 cam_resnet.py --method layercam 
# write into a log file 
python3 cam_resnet.py --method layercam > log
# use cuda
python3 cam_resnet.py --method layercam --use-cuda
```

## Dataset V2 
If you want to run model trained by dataset V2, remember to revise num_classes.

Take resnet18 for example, you need to change ``package/FL/resnet.py`` `line: 72 ` into:

```python
def __init__(self, block, num_blocks, num_classes=35):
```

## CUDA
Some of the model are trained by ``CUDA: 1``, if you only have ``cuda: 0``, you may get error when loading model.

Take ``cam_V2.py line: 93`` for example, you need to add `map_location='cuda:0'`:
```python
global_model.load_state_dict(torch.load(PATH, map_location='cuda:0'))
```


## Models

| file                       | trigger size | addversry ratio | dataset | trigger position | 
|----------------------------|--------------|-----------------|---------|------------------|
| regnet_15_0.2_V1_start.pth | 15%          | 0.2             | V1      | start            |
| regnet_15_0.3_V1_1-4.pth   | 15%          | 0.3             | V1      | 1/4              |
| regnet_15_0.3_V1_3-4.pth   | 15%          | 0.3             | V1      | 3/4              |
| regnet_15_0.3_V1_mid.pth   | 15%          | 0.3             | V1      | mid              |
| regnet_15_0.3_V1_start.pth | 15%          | 0.3             | V1      | start            |
| regnet_V1_clean.pth        | -            | 0.0             | V1      | -                |
| resnet_5_0.2_V1_start.pth  |  5%          | 0.2             | V1      | start            |
| resnet_10_0.2_V1_start.pth | 10%          | 0.2             | V1      | start            |
| resnet_15_0.2_V1_start.pth | 15%          | 0.2             | V1      | start            |
| resnet_15_0.2_V2_start.pth | 15%          | 0.2             | V2      | start            |
| resnet_15_0.3_V1_start.pth | 15%          | 0.3             | V1      | start            |
| resnet_15_0.4_V1_start.pth | 15%          | 0.4             | V1      | start            |
| resnet_25_0.2_V1_start.pth | 25%          | 0.2             | V1      | start            |
| resnet_50_0.2_V1_start.pth | 50%          | 0.2             | V1      | start            |
| resnet_75_0.2_V1_start.pth | 75%          | 0.2             | V1      | start            |
| resnet_V1_clean.pth        | -            | 0.0             | V1      | -                |
| resnet_V2_clean.pth        | -            | 0.0             | V2      | -                |
| resnext_15_0.2_V1_start.pth| 15%          | 0.2             | V1      | start            |
| resnext_V1_clean.pth       | -            | 0.0             | V1      | -                |

