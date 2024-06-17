# Deep Learning for Computer Vision Package


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nyandwi/deep-computer-vision)


Visual data(such as images, video) are everywhere. Rougly, millions of images and videos are generated everyday. For instance, everyday, 95 million photos and 720.000 hours of videos are uploaded on Instagram and YouTube respectively. Computer Vision equipes us with different ways to process and understand such massive datasets in different areas of applications such as self-driving cars, medicine, streaming websites, etc...

Deep Learning has revolutionized Computer Vision. Thanks to the latest advances in deep learning techniques, frameworks, and algorithms, it's now possible to build, train, and evaluate visual recognition systems on real-world datasets.

This is Deep Learning for Computer Vision Package. It is designed exactly like Complete Machine Learning Package, but it's even better. It covers foundations of computer vision and deep learning, state-of-the-arts visual architectures(such as ConvNets and Vision Transformers), various Computer Vision tasks(such as image classification, object detection and segmentation), tips and tricks for training and analyzing visual recognition systems.

<!-- For used tools, check tools overview. For further Computer Vision learning resources, check further resources page. -->

#### Release Notes
* [16 June 2024] Releasing the first draft after several months on hold due to grad school. Depending on the interests, we can add more remaining chapters, add more resources, and build navigable webpage around the materials. Feedback welcome!
* [2022] V1 Release month: Aug-Oct 2022, or Sept 26(would be same as ML package). LATE BUT OKAY!
* [25 Feb 2023] No progress has been made in almost 4 months(a year tbh) but okay. This repo is a proof that 1% percent of the whole work was done and 99% remaining can be done. Don't compromise on the quality for time.



## Outline

### PART 1 - Foundations of Computer Vision and Deep Learning

#### 1. Introduction to Computer Vision | [Notebook](./1-introduction/1_intro_to_computer_vision.ipynb)

- What is Computer Vision
- Industrial Applications of Computer Vision
- History of Computer Vision
- Typical Computer Vision Tasks
- Computer Vision Systems Challenges
- Computer Vision Tools Landscape

#### 2. Basics of Image Processing | [Notebook](./1-introduction/2_image_processing.ipynb)

* Intro to Image Processing
* Image Color Channels
* Image Color Spaces Conversion
* Image Adjustments
* Geometric Transformation: Resizing, Cropping, Flipping, Rotating, Transposing
* Image Kernels and Convolution
* Drawing Bounding Boxes On Image
* Image Histograms and Histograms Equalization

#### 3. Fundamentals of Deep Learning | [Notebook](./1-introduction/3_intro_deep_learning.ipynb)
* What is Neural Networks and Deep Learning?
* The History of Neural Networks
* Types of Neural Networks Architectures
* Basics of Training Neural Networks
    * Activation Functions
    * Loss and Optimization Functions
    * Learning Rate Scheduling
    * Regularization
    * Neural Networks Training Practices
    * Challenges of Training Neural Networks
    * Deep Learning Accelerators and Frameworks

* Neural Networks and Biological Networks

#### 4. Image Classification with Artificial Neural Networks | [Notebook](./1-introduction/4_image_classification.ipynb)

* What is Image Classification?
* Types of Image Classification Problems
* Image Classification Datasets
* Typical Hyperparameters for Image Classification Problems
* Image Classification in Practice

#### 5. Modern Data Augmentation Techniques(TBD)

### PART 2 - Convolutional Neural Networks

#### 6. Intro to Convolutional Neural Networks | [Notebook](./2-convnets/1-introduction-to-convnets.ipynb)

* Fully Connected Layers
* Typical Components of ConvNets
    * Convolutional Layer
    * Pooling Layer
    * Fully Connected Layer
* Batch Normalization for ConvNets
* Other Types of Convolution Layers
* Coding ConvNets: Cifar10 Classification

#### 6. Modern ConvNets Architectures | [Notebook](./3-convnets-architectures/)

* AlexNet - Deep Convolutional Neural Networks
  
* VGG - Very Deep Convolutional Networks for Large Scale Image Recognition

* GoogLeNet(Inceptionv1) - Going Deeper with Convolutions

* ResNet - Deep Residual Learning for Image Recognition

* ResNeXt - Aggregated Residual Transformations for Deep Neural Networks

* Xception - Deep Learning with Depthwise Separable Convolutions

* DenseNet - Densely Connected Convolutional Neural Networks

* MobileNetV1 - Efficient Convolutional Neural Networks for Mobile Vision Applications

* MobileNetV2 - Inverted Residuals and Linear Bottlenecks

* EfficientNet - Rethinking Model Scaling for Convolutional Neural Networks

* RegNet - Designing Network Design Spaces

* ConvMixer - Patches are All You Need?

* ConvNeXt - A ConvNet for the 2020s

#### 8. Choosing a ConvNet Architecture - Size Vs Accuracy(TBD)

#### 9. Transfer Learning with Pretrained ConvNets(TBD)
#### 10. Visualizing ConvNets and Generating Images(TBD)

### PART 4 - Object Detection
#### 11. Introduction to Object Detection | [Notebook](./4-object-detection-segmentation/1-intro-object-detection.ipynb)
* What is Object Detection
* Classification Vs Detection Vs Segmentation
* Applications of Object Detection
* Modern Object Detectors
* Object Detection Metrics
* Object Detection Datasets and Tools Landscape
* The Challenges of Object Detection

#### 12. Object Detection with Detectron2 | [Notebook](./4-object-detection-segmentation/2-intro-detectron2.ipynb)

* Overview of Detectron2
* Detectron2 Model Zoo
* Performing Inference with Pretrained Detector

#### 13. Modern Object Detectors in Practice | [Notebook](./4-object-detection-segmentation/3-vehicle-detection.ipynb)
* Vehicle Detection with Faster R-CNN

### PART 5 - Pixel-Level Recognition

#### 14. Introduction to Pixel Level Recognition | [Notebook](./4-object-detection-segmentation/4-intro-segmentation.ipynb)

* Pixel-level Recognition: Overview
* Semantic Segmentation
* Instance Segmentation
* Panoptic Segmentation
* Pose Estimation
* Modern Applications of Segmentation Tasks

#### 15. Pixel-Level Recognition In Practice
* Semantic Segmentation with DeepLabv3+ and PointRend | [Notebook](./4-object-detection-segmentation/5-semantic-segmentation.ipynb)
* Instance Segmentation with Mask R-CNNN | [Notebook](./4-object-detection-segmentation/6-instance-segmentation.ipynb)
* Panatonic Segmentation with Panoptic FPN | [Notebook](./4-object-detection-segmentation/7-panoptic-segmentation.ipynb)
* Human Pose Estimation with Keypoint R-CNN | [Notebook](./4-object-detection-segmentation/8-human-pose-estimation.ipynb)

### PART 6 - Recurrent Networks, Attention, and Transformers

#### 16. Recurrent Neural Networks | [Notebook](./4-rnns-attention/1-recurrent-networks.ipynb)
* Introduction to Recurrent Neural Networks
* The Downsides of Vanilla RNNs
* Other Recurrent Networks: LSTMs & GRUs
* Recurrent Networks for Computer Vision

#### 17. Attention and Transformer | [Notebook](./4-rnns-attention/2-attention-transformer.ipynb)

* The Downsides of Recurrent Networks and Motivation of Transformers
* Transformer Architecture
    * Attention and Multi-Head Attention
    * Embedding and Positional Encoding Layers
    * Residual Connections, Layer Normalization, and Dropout
    * Linear and Softmax Layers
    * Encoder and Decoder
* Advantages and Disadvantages of Self-Attention
* Implementations of Transformer
* Evolution of Large Language Transformer Models
* Transformers Beyond NLP

### PART 7 - Vision Transformers for Visual Recognition
#### 18. Introduction to Vision Transformers | [Notebook](./5-vision-transformers/1-intro-vision-transformer.ipynb)

* Vision Transformer(ViT)
  * Introduction to ViT
  * Vision Transformer (ViT) Architecture
  * Comparison of ViTs and ConvNets(ResNets)
  * Scaling Vision Transformers
  * Visualizing the Internal Representations of Vision Transformer
* Training and Improving Vision Transformers
  * Improving ViTs with Regularization and Augmentations
  * Improving ViTs with Knowledge Distillation
* Vision Transformers Beyond Image Classification
* Implementations of ViTs

#### 19. Vision Transformers: Case Studies

- DEtection TRansformer(DETR) for Object Detection | [Notebook](./5-vision-transformers/2-detr.ipynb)
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | [Notebook](./5-vision-transformers/3-swin-transformer.ipynb)
- Vision Transformers for Mobile Applications | [Notebook](./5-vision-transformers/5-vision-transformers-mobile.ipynb)
- MaskFormer, Mask2Former, Uniformer(TBD)

### PART 8 - Deep Generative Networks - Upcoming
<!-- To cover:
* Auto-regressive generative networks
* Variation AutoEncoders(VAE)
* Generative Adversarial Networks(GANs)
* Diffusion Models
* Image Generation State of the Arts: Text-Image generation
* Image Generation State of the Arts: Text-Video Generation -->
#### 20. Introduction to Deep Generative Networks | [Notebook](./6-generative-networks/1-intro-generative-networks.ipynb)

* Introducing Generative Models: Supervised and Unsupervised Learning, Generative and Discriminative Models
* Applications of Generative Models
* Recent Breakthroughs in Generative modelling

#### 21. Deep Autoregressive Generative Networks | [Notebook](./6-generative-networks/2-autoregressive-generative-networks.ipynb)

* Introduction to autoregressive models
* Auto-regressive models architectures
  * Pixel Recurrent Neural Networks - PixelRNN
  * Pixel Convolutional Neural Networks - PixelCNN
  * Image Transformer
  * Image GPT - Generative Pretraining from Pixels
* Advantages and disadvantages of autoregressive models

#### 22. Variation AutoEncoders(VAE), Ongoing | [Notebook](./6-generative-networks/3-variation-auto-encoders.ipynb)






