# High-Order-Discriminant-Analysis
Exploratory analysis of utilising High Order Discriminant Analysis for classification tasks

## Requirements

To install the required dependencies, run:
pip install -r requirements.txt

## Background

The reference paper can be found at this link: https://www.jstage.jst.go.jp/article/nolta/1/1/1_1_37/_article .

High order discriminant analysis (HODA) is especially useful to decompose high dimensional data structure, pattern recognition, and to reduce the risk of overfitting for image segmentation/detection tasks. Reducing the original raw data tensor to a lower dimensional representation is also expected to produce significant time savings for model fitting. 

## Overview

Multidimensional tensors are created representing the training data and test data.

In this task, each train/test tensor is 3-dimensional, with each frontal slice representing each image (128x128). The dimension of the train tensor is 128x128x840, where 840 = 20 (number of classes) x 42 (number of training samples per class). The dimension of the test tensor is 128x128x600, where 600 = 20 (number of classes) x 30 (number of training samples per class).

Using the raw train tensor, HODA finds the core train tensor of a lower dimension, along with the set of basis matrices representing its feature space. The basis matrices found are used along with the raw test tensor to extract its lower dimensional representation (getting the core test tensor). The core train tensor is reshaped to a sample/feature matrix and used to train a multinomial SVM classifier. Test accuracy of the model is reported on the core test tensor, also reshaped to a sample/feature matrix.

## Notebook Instructions

1. Specify the folder path as instructed.
2. Ensure the images are according to the specified naming convention so that they are parsed correctly. Image files have been stored with naming 'objX_Y.jpg, where X and Y refer to the object number (class) and sample (sample number within class). There are 20 classes, and within each class, 72 samples. The first 42 samples within each class are used for training, while the remaining 30 samples for test.
3. Alter the attributes for the image folder as needed under folder_attribute.
4. Run the notebook.

## Result Summary

HODA decomposes the raw train tensor from a dimension of 128x128x840 to 10x10x840. At convergence, the between-class/within class scatter matrix ratio (across multiple runs) is noted to be > 8, indicative of separation achieved. The SVM classifier with RBF kernel performs relatively well, with an accuracy exceeding 97%. The reduction in model fitting time with the compressed tensor to a fraction of the case when raw tensor is used is a significant plus, and will help mitigate bottlenecks for hyperparameter tuning for large datasets in large scale applications.
