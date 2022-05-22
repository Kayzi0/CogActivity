# CogActivity
Activity Monitoring and pattern recognition exercise on the cognitive village dataset

## Quickstart : Running the model

To run the model:
  - get the data from https://cloud.imi.uni-luebeck.de/s/PNYNMxRcdid4Lbe
  - rename the folder containing the data to data_raw
  - run `preprocessing.ipynb`
  - if you want to train a network yourself, run `network_train.ipynb`. You can adjust the network you want to use (either `ConvNet()` for a fully convolutional network or `ResNet()` for a residual network) and the `epochs`
  - run `network_test.ipynb`, choosing the correct network number ( see [trained_models](#trained-models) for more info on the pretrained networks)
## Approach
We use a fairly general-purpose deep-learning approach to perform the classification. Research has shown that CNNs are very good at extracting features themselves, so very little preprocessing and no feature engineering was needed.

## Preparing the data
(More info on the individual steps can be found in the preprocessing.ipynb)

The raw data is already seperated into training and testing data, so no manual seperating is necessary. As the individual observations are all four seconds long, but sampled at different frequencies, the data for each individual sensor is of shape `Nx4*Frequencyx3`. 

In order to combine the data into one dataset, manipulating the data to have the same shape is necessary. To achieve this, we resampled all of the sensor data to 200Hz. As some of the data has a lower, and some of the data has a higher frequency, some had to be unsampled, while some had to be decimated. We deciced on a frequency in the middle of the obsvervations in order to minimise the loss of the resampling process.

Secondly, the data was denoised using a [Wiener-Filter](https://en.wikipedia.org/wiki/Wiener_filter).

More visualisations are given in `preprocessing.ipynb`, but in short, we tried both resampling -> denoising and denoising -> resampling and the former proved to yield better results.

As the final step, the data is concatenated into one dataset for training and testing each, and the data is reshaped from `Nx800x8x3` to `Nx800x24` through seperating the channels in order to make 1d-convolutions possible.
Only 8 of the 9 sensors are used, as the accelerometer of the smartglasses has around 2/3 of its data missing, so interpolating from existing data would not be feasible.

## The Models
We tried two different models:
- a three-layer convolutional network with a fully connected layer as a classifier
- a four-block residual network with a fully connected layer as a classifier

## Training
For training, an 80-20 split into training and validation data was used. The training loop is a fairly run of the mill PyTorch training loop. 
For the loss, a CrossEntropyLoss with inverse square adjusted weights was used, as this is most suitable for a multi-class classification.
The optimizer is a standard Adam optimizer with initial `lr = 0.001` and a learning rate scheduler `ReduceLROnPlateu`, which dynamically adjusts the learning rate once the loss hits a plateau.

## Peformance
Due to resource constraints we were not able to train a network for a longer amount of time. However, our best-performing network, a ResNet trained for 301 epochs gave us 64% test accuracy. The loss also seems fairly stable at this point in training, so it is highly likely that the performance would not increase significantly with further training. 
<a name = "models"></a>
## Trained models
- net_0.pt: ConvNet, 61% test accuracy, 66% MAP, 60% f1 score, 100 epochs
- net_1.pt: ResNet, 64% test accuracy, 69% MAP, 64% f1 score, 301 epochs
- net_2.pt: ResNet, 63% test accuracy, 68% MAP, 63% f1 score, 100 epochs
- net_3.pt: ConvNet, 62% test accuracy, 66,5% MAP, f1: 62% f1 score, 300 epochs
