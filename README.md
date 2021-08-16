# gender-classification
WE are making a gender classification model with the help of cnn,convolutional neural networks

Using this corpus we will train our model bases on neural networks to get a model which can classify images (containing faces) as male and females images respectively.
.
.
dowmloading the dataset- We will start by downloading the dataset in our jupyter notebook. Now this can be easily done using “opendatasets”  library in python. We will import the library and will use its “download” to download the dataset by passing the link of the dataset as a parameter in the download function
.
Displaying the image - Imshow” function from Matplotlib library of python will be used to display images in jpg or other image formats in our jupyter notebook
.
Normalizing and augmenting the data - We will normalize the image tensors by subtracting the mean and dividing by the standard deviation across each channel. As a result, the mean of the data across each channel is 0, and standard deviation is 1. Normalizing the data prevents the values from any one channel from disproportionately affecting the losses and gradients while training, simply by having a higher or wider range of values than others.
We will apply randomly chosen transformations while loading images from the training dataset. Specifically, we will pad each image by 4 pixels, and then take a random crop of size 32 x 32 pixels, and then flip the image horizontally with a 50% probability. Since the transformation will be applied randomly and dynamically each time a particular image is loaded, the model sees slightly different images in each epoch of training, which allows it generalize better.
.
Setting up the GPU- GPU, graphics processing unit is a hardware.As the sizes of our models and datasets increase, we need to use GPUs to train our models within a reasonable amount of time. GPUs contain hundreds of cores optimized for performing expensive matrix operations on floating-point numbers quickly, making them ideal for training deep neural networks.  
We are doing multiple large matrix operations in our model, so to make these operations faster we will set up the GPU for training our model.
we will define a couple of helper functions (get_default_device & to_device) and a helper class DeviceDataLoader to move our model & data to the GPU as required.
.
Defining the model - Now we will define the model by extending an ImageClassificationBase class which contains helper  methods for training & validation.

> The training_step function generates the predictions and calculates the loss using the cross entropy function.

>the validation_step  function generates the predictions calculates the loss using cross entropy and calculates the accuracy using the acuuracy function which we defined using the torch.tensor  functions it calculates the accuracy of the model's prediction on an batch of inputs.

>validation_epoch_end function combines the accuracy and loss of all the batches by taking their average and gives us an overall validation loss and accuracy

>epoch_end  function prints all the results the training loss, validation loss and validation accuracy .
.
Applying convolutional neural networks-We will use a convolutional neural network, using the nn.Conv2d class from PyTorch.

The 2D convolution is a fairly simple operation we start with a kernel, which is simply a small matrix of weights. This kernel “slides” over the 2D input data, performing an element wise multiplication with the part of the input it is currently on, and then summing up the results into a single output pixel
There are certain advantages offered by convolutional layers when working with image data:
>Fewer parameters: A small set of parameters (the kernel) is used to calculate outputs of the entire image, so the model has much fewer parameters compared to a fully connected layer.
>Sparsity of connections: In each layer, each output element only depends on a small number of input elements, 
>Parameter sharing and spatial invariance: The features learned by a kernel in one part of the image can be used to detect similar pattern in a different part of another image.
We will also use a max-pooling layers to progressively decrease the height & width of the output tensors from each convolutional layer. 
Max-pool takes a block of 2x2 matrix and takes out the maximum element from it .
The Conv2d layer transforms a 3-channel image to a 16-channel feature map, and the MaxPool2d layer halves the height and width. The feature map gets smaller as we add more layers, until we are finally left with a small feature map, which can be flattened into a vector. We can then add some fully connected layers at the end to get vector of size 2 for each image.
.
RESNET- We will now add residual blocks to our convolutional neural network.This  residual block adds the original input back to the output feature map obtained by passing the input through one or more convolutional layers.

This seeming small change produces a drastic improvement in the performance of the model. Also, after each convolutional layer, we'll add a batch normalization layer, which normalizes the outputs of the previous layer.
adding some improvements- Before we train the model, we're going to make a bunch of small but important improvements to our fit function:

Learning rate scheduling: Instead of using a fixed learning rate, we will use a learning rate scheduler, which will change the learning rate after every batch of training. There are many strategies for varying the learning rate during training, and the one we'll use is called the "One Cycle Learning Rate Policy", which involves starting with a low learning rate, gradually increasing it batch-by-batch to a high learning rate for about 30% of epochs, then gradually decreasing it to a very low value for the remaining epochs
Weight decay: We also use weight decay, which is yet another regularization technique which prevents the weights from becoming too large by adding an additional term to the loss function.
Gradient clipping: Apart from the layer weights and outputs, it also helpful to limit the values of gradients to a small range to prevent undesirable changes in parameters due to large gradient values. This simple yet effective technique is called gradient clipping.
.
Training the model- We are now ready to train our model. We will do this by using the evaluate function which we created. And instead of SGD (stochastic gradient descent), we'll use the Adam optimizer which uses techniques like momentum and adaptive learning rates for faster training.
After training our model reached at maximum 98% percent of accuracy.
Now we can test our model on single images to see if it predicts the correct gender or not.
