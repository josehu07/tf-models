# For Waistband Profiling:

Take MNIST model at `official/legacy/image_classification/mnist_main.py` for example. Create a symbolic link of the file under submodule
`waistband/profile/profile_wrappers.py` to the same folder alongside the model's main script:

```python
cd official/legacy/image_classification/
ln -s ../../../waistband/profile/profile_wrappers.py profile_wrappers.py
```

To enable dataset size profiling, add the following import after all other imports:

```python
# mnist_main.py
from profile_wrappers import *
```

Then, wrap around the dataset loading code snippet with the following two functions:

```python
# mnist_main.py
def run(flags_obj, datasets_override=None, strategy_override=None):
  ...

  profile_wrappers_on()

  mnist = tfds.builder('mnist', data_dir=flags_obj.data_dir)
  if flags_obj.download:
    mnist.download_and_prepare()

  mnist_train, mnist_test = datasets_override or mnist.as_dataset(
      split=['train', 'test'],
      decoders={'image': decode_image()},  # pylint: disable=no-value-for-parameter
      as_supervised=True)
  train_input_dataset = mnist_train.cache().repeat().shuffle(
      buffer_size=50000).batch(flags_obj.batch_size)
  eval_input_dataset = mnist_test.cache().repeat().batch(flags_obj.batch_size)

  profile_wrappers_off()

  ...
```

Finally, run the model (can be without any training epoch -- just ensure that the dataset loading part is executed).

### MNIST

Soft link to wrappers has been added.

```bash
cd official/legacy/image_classification
python3 mnist_main.py \
    --model_dir=mnist_model/ \
    --data_dir=mnist_data/ \
    --train_epochs=0 \
    --distribution_strategy=one_device \
    --num_gpus=0 \
    [--download]  # do this for the first run to download data
```

Transformation program appears in `official/legacy/image_classification/mnist_main.py`.

### RetinaNet on TinyCOCO

Soft link to wrappers has been added.

```bash
cd official/legacy/detection
./run-retinanet.sh train
./run-retinanet.sh eval     # this hangs at the end, just kill
```

The TinyCOCO dataset in `tfrecord` format has been included in-place (since the preparation steps are too verbose).

Transformation program appears in `official/legacy/detection/dataloader/input_reader.py`.


<div align="center">
  <img src="https://storage.googleapis.com/tf_model_garden/tf_model_garden_logo.png">
</div>

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)

# Welcome to the Model Garden for TensorFlow

The TensorFlow Model Garden is a repository with a number of different
implementations of state-of-the-art (SOTA) models and modeling solutions for
TensorFlow users. We aim to demonstrate the best practices for modeling so that
TensorFlow users can take full advantage of TensorFlow for their research and
product development.

To improve the transparency and reproducibility of our models, training logs on
[TensorBoard.dev](https://tensorboard.dev) are also provided for models to the
extent possible though not all models are suitable.

| Directory | Description |
|-----------|-------------|
| [official](official) | • A collection of example implementations for SOTA models using the latest TensorFlow 2's high-level APIs<br />• Officially maintained, supported, and kept up to date with the latest TensorFlow 2 APIs by TensorFlow<br />• Reasonably optimized for fast performance while still being easy to read |
| [research](research) | • A collection of research model implementations in TensorFlow 1 or 2 by researchers<br />• Maintained and supported by researchers |
| [community](community) | • A curated list of the GitHub repositories with machine learning models and implementations powered by TensorFlow 2 |
| [orbit](orbit) | • A flexible and lightweight library that users can easily use or fork when writing customized training loop code in TensorFlow 2.x. It seamlessly integrates with `tf.distribute` and supports running on different device types (CPU, GPU, and TPU). |

## Installation

To install the current release of tensorflow-models, please follow any one of the methods described below.

#### Method 1: Install the TensorFlow Model Garden pip package

<details>

**tf-models-official** is the stable Model Garden package.
pip will install all models and dependencies automatically.

```shell
pip3 install tf-models-official
```

If you are using nlp packages, please also install **tensorflow-text**:

```shell
pip3 install tensorflow-text
```

Please check out our [example](https://github.com/tensorflow/text/blob/master/docs/tutorials/fine_tune_bert.ipynb)
to learn how to use a PIP package.

Note that **tf-models-official** may not include the latest changes in this
github repo. To include latest changes, you may install **tf-models-nightly**,
which is the nightly Model Garden package created daily automatically.

```shell
pip3 install tf-models-nightly
```

If you are using `nlp` packages, please also install tensorflow-text-nightly

```shell
pip3 install tensorflow-text-nightly
```
</details>


#### Method 2: Clone the source

<details>

1. Clone the GitHub repository:

```shell
git clone https://github.com/tensorflow/models.git
```

2. Add the top-level ***/models*** folder to the Python path.

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/models
```

If you are using a Colab notebook, please set the Python path with os.environ.

```python
import os
os.environ['PYTHONPATH'] += ":/path/to/models"
```

3. Install other dependencies

```shell
pip3 install --user -r official/requirements.txt
```

Finally, if you are using nlp packages, please also install
**tensorflow-text-nightly**:

```shell
pip3 install tensorflow-text-nightly
```

</details>


## Announcements

Please check [this page](https://github.com/tensorflow/models/wiki/Announcements) for recent announcements.

## Contributions

[![help wanted:paper implementation](https://img.shields.io/github/issues/tensorflow/models/help%20wanted%3Apaper%20implementation)](https://github.com/tensorflow/models/labels/help%20wanted%3Apaper%20implementation)

If you want to contribute, please review the [contribution guidelines](https://github.com/tensorflow/models/wiki/How-to-contribute).

## License

[Apache License 2.0](LICENSE)

## Citing TensorFlow Model Garden

If you use TensorFlow Model Garden in your research, please cite this repository.

```
@misc{tensorflowmodelgarden2020,
  author = {Hongkun Yu, Chen Chen, Xianzhi Du, Yeqing Li, Abdullah Rashwan, Le Hou, Pengchong Jin, Fan Yang,
            Frederick Liu, Jaeyoun Kim, and Jing Li},
  title = {{TensorFlow Model Garden}},
  howpublished = {\url{https://github.com/tensorflow/models}},
  year = {2020}
}
```
