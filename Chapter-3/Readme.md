### Classifying Movie Reviews - Problem 1
The dataset contains 50k polarised movie reviews. 25k positive and 25k negative reviews .
#### Why should we split data into train and test?
- Models that perform well on the test data needn't perform well on the training data 
- The model could just end up memorising the training instances
- Our aim should be to identify a model that generalizes well
- So having the train and test as separate sets helps us evaluate the model better 

#### Reading Dataset using keras datasets
- keras.datasets import imdb 
- contains train and test data
- 50 % data is train and 50 % is test 
- 0 indicates a negative review and 1 indicates a positive review
- num_words - specifies the number of words in the vocab of the dataset; it picks the top 10k words

#### Checking max word index in the data 
 - To see if we have any word above the specified word index
 - The max is 9k

#### Decoding the sequences 
- imdb has a parameter called word index which can be used to decode a sequence 
- in the word indices 
  1. 0 - reserved for padding 
  2. 1 - start of a sequence
  3. 2 - reserved for out of vocabulary tokens

#### Data Preparation 
- The sequences we have can't act as an input to the neural network, because they can't be converted to tensors implicitly
- There are two ways to convert them 
  - pad the interger seqs with 0 and then feed this to an embedding layer 
  - one hot encode all the words into a vector of 10k features ;
    - this is what will be followed for this tutorial 


#### Building a network 
- A neural network with relu activation in the hidden layers and sigmoid in the output layer should be built
- The reason for this config is we have vector input and a scalar output
- Relu : max(0,X) - zeros out the negative part of the activation
- Sigmoid - maps the input into a space of 0 and 1
- There are methods to choose the number of layers and the number of neurons in each layer 
  - The more complex the network is it is susceptible to over-fitting


#### Why do we use activation functions?
- Without the activation - we are just doing a bunch of linear transformation to the data 
- If we don't use activation functions we won't be able to map out the input to complex non-linear functions 


#### Loss Selection 
- the selection of the appropriate loss function determines the ease and quality of training 
- binary classification - binary_crossentropy - Finds the distance between the ground truth and the probabilities predicted 
- We can also use MSE, but it will not be convex so gradient decent will not converge at the global minimum
- Custom lossfunctions can also be used


#### Optimizer 
- we are using rmsprop which is a slightly modified version of regular gradient decent  

#### Metrics to minitor while training 
- Accuracy and loss

#### Validation data 
- This is used to check if the model overfits in this scenario 
- It can also be used for tuning hyperparamets which is not covered here 

#### Model Training
- Epochs - no of times the model has to iterate over the training data 
- The training and validation cures have been plot to understand if the model is generalizing or just memorising
- We can observe that the model after 3 epochs starts to overfit 

