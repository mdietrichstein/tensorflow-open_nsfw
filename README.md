# Tensorflow Implementation of Yahoo's Open NSFW Model

This repository contains an implementation of [Yahoo's Open NSFW Classifier](https://github.com/yahoo/open_nsfw) rewritten in tensorflow.

The original caffe weights have been extracted using [Caffe to TensorFlow](https://github.com/ethereon/caffe-tensorflow). You can find them at `data/open_nsfw-weights.npy`.

## Prerequisites

All code should be compatible with `Python 3.6` and `Tensorflow 1.0.0`. The model implementation can be found in `model.py`.

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
  -t {tensor,base64_jpeg}, --input_type {tensor,base64_jpeg}
                        input type
```

__-l/--image-loader__

The classification tool supports two different image loading mechanisms. 

* `yahoo` (default) tries to replicate the image loading mechanism used by the original caffe implementation, differs a bit though. See __Caveats__ below.
* `tensorflow` is an image loader which uses tensorflow  api's exclusively (no dependencies on `PIL`, `skimage`, etc.).

__Note:__ Classification results may vary depending on the selected image loader!

__-t/--input_type__

Determines if the model internally uses a float tensor (`tensor` - `[None, 224, 224, 3]` - default) or a base64 encoded string tensor (`base64_jpeg` - `[None, ]`) as input. Should not have an effect on the classification.


### Tools

The `tools` folder contains some utility scripts to test the model.

__export_model.py__

Exports the model using the standard tensorflow export api (`SavedModel`). The export can be used to deploy the model on [Google Cloud ML Engine](https://cloud.google.com/ml-engine/docs/concepts/prediction-overview), [Tensorflow Serving]() or on mobile (haven't tried that one yet).

__create_predict_request.py__

Takes an input image and spits out an json file suitable for prediction requests to a Open NSFW Model deployed on [Google Cloud ML Engine](https://cloud.google.com/ml-engine/docs/concepts/prediction-overview) (`gcloud ml-engine predict`).

### Caveats

#### Image loading differences

The classification results sometimes differ more and sometimes less from the original caffe implementation, depending on the image loader and input image. I haven't been able to figure out the cause for this yet. Any help on this would be appreciated.
