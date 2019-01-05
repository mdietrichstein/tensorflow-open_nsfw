# Tensorflow Implementation of Yahoo's Open NSFW Model

This repository contains an implementation of [Yahoo's Open NSFW Classifier](https://github.com/yahoo/open_nsfw) rewritten in tensorflow.

The original caffe weights have been extracted using [Caffe to TensorFlow](https://github.com/ethereon/caffe-tensorflow). You can find them at `data/open_nsfw-weights.npy`.

## Prerequisites

All code should be compatible with `Python 3.6` and `Tensorflow 1.x` (tested with 1.12). The model implementation can be found in `model.py`.

### Usage

```
> python classify_nsfw.py -m data/open_nsfw-weights.npy test.jpg

Results for 'test.jpg'
	SFW score:	0.9355766177177429
	NSFW score:	0.06442338228225708
```

__Note:__ Currently only jpeg images are supported.

`classify_nsfw.py` accepts some optional parameters you may want to play around with:

```
usage: classify_nsfw.py [-h] -m MODEL_WEIGHTS [-l {yahoo,tensorflow}]
                        [-t {tensor,base64_jpeg}]
                        input_jpeg_file

positional arguments:
  input_file            Path to the input image. Only jpeg images are
                        supported.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_WEIGHTS, --model_weights MODEL_WEIGHTS
                        Path to trained model weights file
  -l {yahoo,tensorflow}, --image_loader {yahoo,tensorflow}
                        image loading mechanism
  -i {tensor,base64_jpeg}, --input_type {tensor,base64_jpeg}
                        input type
```

__-l/--image-loader__

The classification tool supports two different image loading mechanisms. 

* `yahoo` (default) replicates yahoo's original image loading and preprocessing. Use this option if you want the same results as with the original implementation
* `tensorflow` is an image loader which uses tensorflow exclusively (no dependencies on `PIL`, `skimage`, etc.). Tries to replicate the image loading mechanism used by the original caffe implementation, differs a bit though due to different jpeg and resizing implementations. See [this issue](https://github.com/mdietrichstein/tensorflow-open_nsfw/issues/2#issuecomment-346125345) for details.

__Note:__ Classification results may vary depending on the selected image loader!

__-i/--input_type__

Determines if the model internally uses a float tensor (`tensor` - `[None, 224, 224, 3]` - default) or a base64 encoded string tensor (`base64_jpeg` - `[None, ]`) as input. If `base64_jpeg` is used, then the `tensorflow` image loader will be used, regardless of the _-l/--image-loader_ argument.


### Tools

The `tools` folder contains some utility scripts to test the model.

__create_predict_request.py__

Takes an input image and generates a json file suitable for prediction requests to a Open NSFW Model deployed with [Google Cloud ML Engine](https://cloud.google.com/ml-engine/docs/concepts/prediction-overview) (`gcloud ml-engine predict`) or [tensorflow-serving](https://www.tensorflow.org/serving/).


__export_savedmodel.py__

Exports the model using the tensorflow serving export api (`SavedModel`). The export can be used to deploy the model on [Google Cloud ML Engine](https://cloud.google.com/ml-engine/docs/concepts/prediction-overview), [Tensorflow Serving]() or on mobile (haven't tried that one yet).

__export_tflite.py__

Exports the model in [TFLite format](https://www.tensorflow.org/lite/). Use this one if you want to run inference on mobile or IoT devices. Please note that the `base64_jpeg` input type does not work with TFLite since the standard runtime lacks a number of required tensorflow operations.

__export_graph.py__

Exports the tensorflow graph and checkpoint. Freezes and optimizes the graph per default for improved inference and deployment usage (e.g. Android, iOS, etc.). Import the graph with `tf.import_graph_def`.