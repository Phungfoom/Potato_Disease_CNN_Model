**Potato_Disease_CNN_Model**
A CNN model to perform image classification on potatoes to detect diseases. Models’ stability will be tested by adding gateway checks.

2nd step: GPS tracking, live weather, historical data. Gives contextual data as validation for Sobel, Grayscale, and Noise experiments.

Dataset that model is first trained on with reduced noise is from, [Kaggle Diseased Leaves](https://www.kaggle.com/datasets/aarishasifkhan/plantvillage-potato-disease-dataset)

Data set of diseased potato leaves is from [Diseased leaves](https://www.kaggle.com/datasets/nirmalsankalana/potato-leaf-disease-dataset), geographically, photos are taken in Central Java, Indonesia. Weather and GPS data from NASA power API to accurately train the model on what weather conditions caused the disease.

**Stress Test**

**Grayscale vs. RGB Analysis:**

Past Research: RGB yields higher results in detecting infected leaves, with the indicators bring highly color dependent. However, over-reliance on chromatic data can lead to lower accuracy in real-world lighting. When these images are converted to gray scale, the accuracy drops significantly. It brings into consideration a model’s capabilities when it comes to lab pictures vs. field pictures. [Comparison of Let Net Architecture, Inception V1 and Inception V3]( https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1308528/full)

My model: Including grayscale baseline, investigating if color is a 'crutch' for the model. The grayscale test evaluates the models’ capabilities to lighting variance, seeing if the model is only relying on color intensity. RGB and Grayscale will be split into two branches to reproduce results from past research and to train the model to produce more accurate results. [Split RGB and Grayscale]( https://mendel-journal.org/index.php/mendel/article/view/176/175)

Sobel Edge: Grad-CAM to Sobel Edges, focuses on edges of legions rather than whole leaf. 
Past research: Uses 'edge-enhances' CNN (EEDB-CNN) show classification project going up but blur the boundary details during pooling. [ResNet and MobileNet edge feature degradation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12777044/)

My model: Testing Sobel Edge separately, 'Edge-Aware Feature Extraction'. One branch performs standard convolutions for color/texture, and the 2nd branch uses Sobel to maintain high-frequency structural data that pooling usually struggles with. Grad_CAM heatmaps follow contours of the disease rather than the contrast of background noise to identify plants. 
Gaussian Noise: 

Past Research: FL-Efficient Net over previous EfficientNet-BO with focal loss, noise augmentation to ensure models can handle complex settings and images where camera quality is poor.  Research has found that adding Gaussian Noise during training prevents the model from overfitting to better quality lab images. [FL-Efficient Noise]( https://www.researchgate.net/publication/362967440_Research_on_plant_disease_identification_based_on_CNN)

My model: Testing signal-to noise-ratio levels and well as a fast gradient sign method to see if adding invisible pixels confuses the model. This aids in simulating poor camera quality photos to prevent overfitting. 

**Mutli Input CNN:**

**Branch A:** Process raw RGB data to find color/shape patterns. It focuses on color and hue shifts.
**Branch B:** Process the Sobel/Texture maps to focus purely on structural decay. Any irregular edges, fuzz or rings forming on leaves.
These two branches merge at the end to make a final decision.
**Evaluation Metrics:**
Confusion Matrix: Heatmap showing which disease the model confuses
Precision Recall Curves: Tradeoff between being careful/thorough
F1-Score: For unbalanced dataset.

**Goal:** The goal of the project is to create a CNN model that does not over rely on lab quality dependencies to classify diseases on potato leaves. It forces the model to learn the biological structure of the disease. By using a dual branch feature fusion approach, the model processes standard RGB along with Sobel edges and Gaussian noise increase model understanding of the disease to identify the consistent structural degradation of the leaves caused by the disease. 
To improve diagnostic accuracy, the model uses environmental data from NASA power API such as local humidity and temp to give probability reasoning, validating the model at the end. 

**Importance:** AI models for plant disease detection work well in lab environment since real-world background noise is no longer captured on camera. In real world farms, with other variables such as low-quality cameras, messy backgrounds, and varied lighting, models have a harder time classifying images. This model is a more resilient model that can survive real world stressors. 

# Getting Started

* How to use repository

## Dependencies

*libraries, packages and otheres. Library versions if needed

# Usage

* Examples of how project can be used. 
* Screenshots, code examples and demos.
* Resource links

# Contributing

# License

* Might put this seperately

# Authors

# Ackowledgements 

**Template set up help** 
https://github.com/catiaspsilva/README-template/blob/main/README.md 