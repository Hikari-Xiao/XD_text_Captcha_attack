# Attention-based one stage attack on text CAPTCHA


## Requirements

1. Install the TensorFlow library ([instructions][TF]). For example:

```
python3 -m venv ~/.tensorflow
source ~/.tensorflow/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow-gpu=1.15
```


2. 16GB of RAM or more; 32GB is recommended.
3. `train.py` works with both CPU and GPU, though using GPU is preferable. It has been tested with a Titan X and with a GTX980.

[TF]: https://www.tensorflow.org/install/
[FSNS]: https://github.com/tensorflow/models/tree/master/research/street

## Dataset

We provide 22 text CAPTCHAs collected from popular websites, including 6 Chinese-based CAPTCHAs and 16 English-based CAPTCHAs.

* Total: around 12GB

The download path is:

```
https://drive.google.com/drive/folders/1tgefqBsNUESpgERgSP21Dn1fadrUrGUf
```

## How to use this code


To train from scratch:

```
python train.py
```

To fine tune the Attention model using a checkpoint:

1. You can download our pre-trained model
```
https://drive.google.com/drive/folders/1hVd5BohwmohNREfRsE1-GcKB7Kk119uu
```
2. use 'saver.restore' in line 82 of train.py
```
python train.py
```

To test on roman-based CAPTCHA:
```
python test.py
```

To test on Chinese-based CAPTCHA:
```
python test_chinese.py
```

## How to use this code with your own dataset


crate tfrecord:

```
python cap_gen_tfrecord.py
or
python chinese_gen_tfrecord.py
```

For Chinese dataset, you also need to make dictionary python file:

```
python create_chineseMap.py
```
