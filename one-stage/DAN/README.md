# DAN-based one stage attack on text CAPTCHA


## Requirements

We recommend you to use [Anaconda](https://www.anaconda.com/) to manage your libraries.

- [Python 2.7](https://www.python.org/) (The data augmentation toolkit does not support python3)
- [PyTorch](https://pytorch.org/) (We have tested 0.4.1 and 1.1.0)
- [TorchVision](https://pypi.org/project/torchvision/)
- [OpenCV](https://opencv.org/)
- [PIL (Pillow)](https://pillow.readthedocs.io/en/stable/#)
- [Colour](https://pypi.org/project/colour/)
- [LMDB](https://pypi.org/project/lmdb/)
- [editdistance](https://pypi.org/project/editdistance/)

Or use [pip](https://pypi.org/project/pip/) to install the libraries. (Maybe the torch is different from the anaconda version. Please check carefully and fix the warnings in training stage if necessary.)

```bash
    pip install -r requirements.txt
```
Besides, a [data augmentation toolkit](https://github.com/Canjie-Luo/Scene-Text-Image-Transformer) is used for handwritten text recognition.

## Data Preparation
### Dataset
We provide 22 text CAPTCHAs collected from popular websites, including 6 Chinese-based CAPTCHAs and 16 English-based CAPTCHAs.

* Total: around 12GB

The download path is:

```
https://drive.google.com/drive/folders/1tgefqBsNUESpgERgSP21Dn1fadrUrGUf?usp=sharing
```
### Scene text

Please convert dataset to **LMDB** format:

```
python read_write_lmdb.py
```

## Training and Testing

You can download our pre-trained model
```
https://drive.google.com/drive/folders/1hVd5BohwmohNREfRsE1-GcKB7Kk119uu
```

Modify the path and config in configuration file (`cfg_captcha.py`):
```
python main.py
```
