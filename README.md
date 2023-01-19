# Simple Style Transfer App

## Project description
This project wraps style transfer models from ONNX Model Zoo into an interactive
web application implemented with FastAPI and Grad.io. This is a minimal version 
of an application based on trained neural networks as the requirements assumed 
2-3 hours of work to complete the task. 

The next steps in developing the project would be:
- adding a script to train new style transfer models for a provided style image,
- extending the model server to load and serve models from a local directory,
- better model loading, right now there's a small unmber of small models so all of them can exsit in memory at a time.
- separating backend and frontend into two projects,
- adding tests and CI pipelines. 

## Tasks implemented
1. Use docker.
2. Use Docker-Compose to create an application with multiple services.
3. Use FastAPI to handle the model.

## Running the project
```commandline
$ sudo docker-compose build
$ sudo docker-compose up 
```

The app should be accessible via a browser at [http://localhost:7860](http://localhost:7860).

## Models and resources
The models used in the project are Fast Neural Style Transfer models described 
in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).
The implementation is based on models from [ONNX Model Zoo](https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style)
and code for usage is taken from the [example](https://github.com/onnx/models/blob/main/vision/style_transfer/fast_neural_style/dependencies/style-transfer-ort.ipynb).

The models from ONNX Zoo were based on [example from pytorch](https://github.com/pytorch/examples/tree/master/fast_neural_style)
and trained on [COCO 2014 dataset](http://cocodataset.org/#download).
