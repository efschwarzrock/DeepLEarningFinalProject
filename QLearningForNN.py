import json

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import random as r
import copy
from saveArchitecture import saveArchitecture

#Chance of taking a random action


epsilon = .5

#How quickly to update Q Values(think how big the jumps are in grad decent)
stepSize = .1

#discounting future rewards(i.e. is it beter to get 5 points now or 50 points in 200 moves, but 50 is uncertain)
discout = .9

#the number of layer types
NUMACTIONS = 4

#the number of termination layers
NumTerminationActions = 1

#number of states we can be in
NUMSTATES = NUMACTIONS - NumTerminationActions

#max number of layers
MAXLAYERS = 5

#Dictionary to store the architecture
saved_architecture = saveArchitecture("a")

current_layer = 0


# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# x_test = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 28 * 28))
# y_test = np.load("fashion_mnist_test_labels.npy")
# x_train = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28 * 28))
# y_train = np.load("fashion_mnist_train_labels.npy")
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# normalize
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Further break training data into train / validation sets (# put 5000 into validation set and keep remaining 55,000 for train)
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')




#the curent state of the model
current_layer = None
layers = None
print(layers)
#The state action pairs
StateActionPairs = np.zeros(shape=(MAXLAYERS+1, NUMSTATES, NUMACTIONS))

#the number of generations/trials run(howmany models have we trained)
gens = 0

#a debugging incremetor
its = 0

#The input size of the next layer
inputSize = 28





parameters = {
    "max_layers": 5,
    "conv_num_filters": [64],
    "conv_filter_sizes": [2],
    "conv_strides": [1],
    "conv_representation_sizes": [0],
    "pooling_sizes_strides": [(2, 1)],
    "pooling_representation_sizes": [0],
    "dense_consecutive": 2,
    "dense_nodes": [128],
    "classes": 10
}


# creates all possible layers from specified parameters
def get_layers():
    # create convolution layers
    convolutions = []
    for i in parameters['conv_num_filters']:
        for j in parameters['conv_filter_sizes']:
            for k in parameters['conv_strides']:
                for l in parameters['conv_representation_sizes']:
                    convolutions.append({'type': 'convolution', 'num_filters': i, 'filter_size': j, 'stride': k,
                                         'representation_size': l})

    # create pooling layers
    poolings = []
    for i in parameters['pooling_sizes_strides']:
        for j in parameters['pooling_representation_sizes']:
            poolings.append({'type': 'pooling', 'pool_size': i[0], 'stride': i[1], 'representation_size': j})

    # create dense layers
    denses = []
    for i in parameters['dense_nodes']:
        denses.append({'type': 'dense', 'nodes': i})

    # create termination layers
    terminations = [{'type': 'softmax', 'nodes': parameters['classes']}]

    layers = {
        "convolution": convolutions,
        "pooling": poolings,
        "dense": denses,
        "termination": terminations
    }
    return layers


# given current layer, randomly choose next layer
# pass in 0 for current if starting

# NOTES
# -currently does not account for representation size
# -no global average pooling
def add_layer(current, random):
    #print("\ncurrent = ", current)
    current_depth = 0
    if current != 0:
        current_depth = current['layer_depth']
    # get possible layers and current layer depth
    layers = get_layers()
    # at beginning, can go to convolution or pooling
    if current == 0:
        next_layers = layers['convolution'] + layers['pooling']
    # if at max depth, return termination state
    elif current_depth == parameters['max_layers'] - 1:
        next_layers = layers['termination']
    # if at convolution, can go to anything
    elif current['type'] == 'convolution':
        next_layers = layers['convolution'] + layers['pooling'] + layers['dense'] + layers['termination']
    # if at pooling, can go to anything but pooling
    elif current['type'] == 'pooling':
        next_layers = layers['convolution'] + layers['dense'] + layers['termination']
    # if at dense and not at max fully connected, can go to another dense or termination
    elif current['type'] == 'dense' and current['consecutive'] != parameters['dense_consecutive']:
        next_layers = layers['dense'] + layers['termination']
    # if at dense and at max fully connected, must go to termination
    elif current['type'] == 'dense' and current['consecutive'] == parameters['dense_consecutive']:
        next_layers = layers['termination']

    # randomly select next layer
    if random:
        rand = r.randint(0, len(next_layers) - 1)
        next_layer = next_layers[rand]
    # select best next layer
    else:
        next_layer = find_best(current, next_layers, current_depth)##############!!!!!!!!!!!!!

    # update layer depth
    #print("start - ", next_layer)
    next_layer['layer_depth'] = current_depth + 1
    #print("end - ", next_layer)
    if current == 0:
        return next_layer

    # update consecutive dense layer if next layer is dense
    if next_layer['type'] == 'dense' and current['type'] != 'dense':
        next_layer['consecutive'] = 1
    if next_layer['type'] == 'dense' and current['type'] == 'dense':
        next_layer['consecutive'] = current['consecutive'] + 1
    return next_layer

#given the current layer and possible next layers, find the best one
def find_best(current, next_layers, depth):
    best_layer = None
    if depth != 0:
        best = -1
        for layer in next_layers:
            current_index = getStateIndex(current)
            next_index = getStateIndex(layer)
            accuracy = StateActionPairs[depth][current_index][next_index]
            if accuracy > best:
                best = accuracy
                best_layer = layer
    else:
        best = -1
        for layer in next_layers:
            current_index = 0
            next_index = getStateIndex(layer)
            accuracy = StateActionPairs[depth][current_index][next_index]
            if accuracy > best:
                best = accuracy
                best_layer = layer


    return best_layer

# generate a random architecture with specified parameters
def generate_architecture():
    layers = [add_layer(0)]
    while (layers[-1]['type'] != 'softmax'):
        layers.append(add_layer(layers[-1]))
    return layers


# given layer dictionary, create TensorFlow layer
def create_tf_layer(layer):
    layer_type = layer['type']
    layer_depth = layer['layer_depth']
    if layer_type == 'convolution' and layer_depth == 1:
        tf_layer = tf.keras.layers.Conv2D(filters=layer['num_filters'], kernel_size=layer['filter_size'],
                                          strides=layer['stride'], padding='same', input_shape=(28, 28, 1))
    elif layer_type == 'pooling' and layer_depth == 1:
        tf_layer = tf.keras.layers.MaxPooling2D(pool_size=layer['pool_size'], strides=layer['stride'],
                                                input_shape=(28, 28, 1))
    elif layer_type == 'convolution':
        tf_layer = tf.keras.layers.Conv2D(filters=layer['num_filters'], kernel_size=layer['filter_size'],
                                          strides=layer['stride'], padding='same')
    elif layer_type == 'pooling':
        tf_layer = tf.keras.layers.MaxPooling2D(pool_size=layer['pool_size'], strides=layer['stride'])
    elif layer_type == 'dense':
        tf_layer = tf.keras.layers.Dense(layer['nodes'], activation='relu')
    elif layer_type == 'softmax':
        tf_layer = tf.keras.layers.Dense(layer['nodes'], activation='softmax')
    return tf_layer


# given list of layer dictionaries, create TensorFlow model
def create_model(architecture):
    model = tf.keras.Sequential()
    for layer in architecture:
        if (layer['type'] == 'dense' and layer['consecutive'] == 1) or layer['type'] == 'softmax':
            model.add(tf.keras.layers.Flatten())
        model.add(create_tf_layer(layer))
    return model


# return validation accuracy of architecture
def evaluate_model(model, batch_size, epochs, x_train, y_train, x_val, y_val):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val),
              callbacks=[checkpointer])

    # return validation accuracy
    model.load_weights('model.weights.best.hdf5')
    score = model.evaluate(x_val, y_val, verbose=0)
    print('\n', 'Accuracy:', score[1])
    return score[1]



######################################################################
######################################################################
######################################################################
######################################################################
######################################################################






#TODO update these function to fit new model
def getConvolutionStateIndex(state):
    '''
    RecepFieldSize = state[1] #{1, 3, 5}
    #the index of the size
    RecepFieldSizeI = 0


    NumRecepFields = state[2] #{64, 128, 256, 512}
    #index of the numRecepFields
    NumRecepFieldsI = 0


    RepSize = state[3] #{(∞, 8], (8, 4], (4, 1]}
    #index of RepSize
    RepSizeI = 0

    if RecepFieldSize == 1:
        RecepFieldSizeI = 0
    elif RecepFieldSize == 3:
        RecepFieldSizeI = 1
    elif RecepFieldSize == 5:
        RecepFieldSizeI = 2

    if NumRecepFields == 64:
        NumRecepFieldsI = 0
    elif NumRecepFields == 128:
        NumRecepFieldsI = 1
    elif NumRecepFields == 256:
        NumRecepFieldsI = 2
    elif NumRecepFields == 512:
        NumRecepFieldsI = 3

    ret = RecepFieldSizeI*12 + NumRecepFieldsI*3 + RepSizeI
    '''
    return 0


def getPoolingStateIndex(state):
    '''
    RecepFieldSize = state[1]  # {(5, 3),(3, 2),(2, 2)}
    # the index of the size
    RecepFieldSizeI = 0


    RepSize = state[3]  # {(∞, 8], (8, 4], (4, 1]}
    # index of RepSize
    RepSizeI = 0

    if RecepFieldSize == 1:
        RecepFieldSizeI = 0
    elif RecepFieldSize == 3:
        RecepFieldSizeI = 1
    elif RecepFieldSize == 5:
        RecepFieldSizeI = 2

    ret = 3*3*4 + RecepFieldSizeI*3 + RepSizeI
    '''
    return 1


def getFullStateIndex(state):
    '''
    #the number of fully conected layers befor this one
    NumConsecFullConec = state[1] # 0, 1

    NumNeurons = state[2] # {512, 256, 128}
    NumNeuronsI = 0

    if NumNeurons == 512:
        NumNeuronsI = 0
    elif NumNeurons == 256:
        NumNeuronsI = 1
    elif NumNeurons == 128:
            NumNeuronsI = 2
    ret = 3*3*4 + 3*3 + NumConsecFullConec*3 + NumNeuronsI
    '''
    return 2


def getTerminationStateIndex(state):
    '''
    TermType = state[1]
    TypeI = 0
    if TermType == "Softmax":
        TypeI = 0
    else:
        TypeI = 1


    ret = 3 * 3 * 4 + 3 * 3 + 3 * 2 + TypeI
    '''
    return 3

#TODO takes in a state or action and returns the index of it in the table
def getStateIndex(state):
    if state['type'] == 'convolution':
        return getConvolutionStateIndex(state)
    elif state['type'] == 'pooling':
        return getPoolingStateIndex(state)
    elif state['type'] == 'dense':
        return getFullStateIndex(state)
    elif state['type'] == 'softmax':
        return getTerminationStateIndex(state)

#TODO takes in an index and returns the state or action it coresponds to
def getIndexState(state):
    pass

'''###########################################################################
##############################################################################
###########################################################################'''

def makeMove(action):
    #TODO update input size, update actions representation size
    global inputSize
    #Pooling
    # new input size = roundup((input -(size-1))/stride))
    if action['type'] == 'pooling':
        inputSize = np.ceil((inputSize - (action['pool_size'] - 1))/action['stride'])
    if inputSize > 7:
        action['representation_size'] = 3
    elif inputSize > 3:
        action['representation_size'] = 2
    else:
        action['representation_size'] = 1
    layers.append(action)
    global current_layer
    current_layer = action
    pass

#return the best next layer based on the state action table
def getBestNextLayer(curr_layer):
    return add_layer(curr_layer, False)

#return a random next layer
def getRandNextLayer(curr_layer):
    return add_layer(curr_layer, True)

#generate next layer
def getNextLayer(curr_layer):
    r = np.random.rand()
    if r > epsilon:
        return getBestNextLayer(curr_layer)
    else:
        return getRandNextLayer(curr_layer)
    
def update(oldState, newState, accuracy):
    if oldState != 0:
        old = StateActionPairs[oldState['layer_depth']][getStateIndex(oldState)][getStateIndex(newState)]
        StateActionPairs[oldState['layer_depth']][getStateIndex(oldState)][getStateIndex(newState)] = old*(1-stepSize) + stepSize*(accuracy + discout*(np.max(StateActionPairs[newState['layer_depth']][getIndexState(newState)])))
    else:
        old = StateActionPairs[0][0][getStateIndex(newState)]
        StateActionPairs[0][0][getStateIndex(newState)] = old * (
                    1 - stepSize) + stepSize * (accuracy + discout * (
            np.max(StateActionPairs[newState['layer_depth']][getIndexState(newState)])))


#return True if we are at a termination layer
#only have softmax for now
def done(layer):
    #TODO update to other thing
    if layer['type'] == 'softmax':
        return True
    return False

#train architecture if we have not already, or get its reward from the table
def trainModel(layers):
    return np.random.rand();
    #if we've already trained this model, get from table
    if str(layers) in saved_architecture:
        accuracy = saved_architecture[str(layers)]
    #otherwise, train the model
    else:
        #why don't we use test data?
        model = create_model(layers)
        accuracy = evaluate_model(model, 128, 1, x_train, y_train, x_valid, y_valid)
    return accuracy

#get the first layer in the architecture
def getFirstLayer():
    r = np.random.rand()

    if r > epsilon:
        return add_layer(0, False);####!!!!!!!!!!!!!!!!!
    else:
        return add_layer(0, True);

def randArchiveUpdate():
    #TODO randomly sample 100 models from the dictionary and update the Q Values(IDK what exactly its supposed to do)
    #its described on page 6 paragraph right under the table

    # get 100 sample from save_architectures
    sampleSize = len(saved_architecture.getKeys())
    if sampleSize >= 100:
        sampleSize = 100

    samples = r.sample(list(saved_architecture.getKeys()), sampleSize)
    # each sample is a long string separated by "|"
    for sample in samples:
        # to get each layer, split by "|", and get the dict structure
        acc = saved_architecture.getValueByString(sample)
        layerList = []
        layers = sample.split("|")

        # construct the model (list of dist)
        for layer in layers:
            layerList += json.loads(layer)

        # apply them to update function
        for l in range(len(layerList), 0):
            if l == len(layerList) - 1:
                update(layerList[l-1], layerList[l], acc)
            elif l == 1:
                update(0, layerList[0], 0)
            else:
                update(layerList[l-1], layerList[l], 0)
    return
    #so what it does is for each model you just call update but in revese order.
    #eaxmple, model [(convo1), (pool), (convo2), (dence), (softmax)], with accuracy "acc"
    # you would call the function update(oldLayer, newLayer, accuracy) in this order
    # update(dence, softmax, acc)
    # update(convo2, dence, 0)
    # update(pool, convo2, 0)
    # update(convo1, pool, 0)
    # update(0, convo1, 0)


# get current epsilon given generation
def epsilonDecay(gens):
    return 1 - gens * 0.0001

def numToStrLength(num, length):
    if num > 999:
        num = 999
    if num < -999:
        num = -999
    s = str(num)
    s = s.center(length)
    return s

def printStateAction(arr):
    for i in range(len(arr)):
        print("layer ", i, " {\n")
        for j in range(len(arr[i])):
            s = ""
            for k in range(len(arr[i][j])):
                s = s + numToStrLength(round(arr[i][j][k]*1000), 3) + ", "
            print("[", s, "]\n")
        print("}\n")

def printLayers(l):
    for i in range(len(l)):
        print(l[i])


'''###########################################################################
##############################################################################
###########################################################################'''

#the curent state of the model
current_layer = getFirstLayer()

layers = []
makeMove(current_layer)







while gens < 10:

    #pick a next layer
    #print("\n\noutside - ", current_layer)
    next_layer = getNextLayer(current_layer)

    #store the old state
    oldState = copy.deepcopy(current_layer)

    #make the move and get the new state
    makeMove(next_layer)
    #print("update - ", current_layer)
    newState = copy.deepcopy(current_layer)


    #Check if we need to train the model
    if done(next_layer):
        printLayers(layers)
        #we need to train so train and archive the model
        accuracy = trainModel(layers)
        saved_architecture.archiveFindings(layers, accuracy)
        print(accuracy, "--------------")
        #update the Q Values and give a reward of accuracy
        #print("befor\n", StateActionPairs)
        #printStateAction(StateActionPairs)
        update(oldState, newState, accuracy);
        #print("after\n", StateActionPairs, "\n\n\n\n")
        #set back to original state and increase generation
        current_layer = getFirstLayer()
        update(0, current_layer, 0)# update the chossing of the first layer
        makeMove(current_layer)
        gens = gens + 1
        inputSize = 28
        #Do stuff the paper says
        randArchiveUpdate()
        #epsilonDecay(gens)
    else:
        #we don't need to so update the QValues with an accuracy of 0 becasue we got no reward since we aren't done
        #update(oldState, move, newState, 0)
        update(current_layer, newState, 0)


    its = its+1

#printStateAction(StateActionPairs)

'''
# generate random architecture and evaluate
architecture = generate_architecture()
model = create_model(architecture)
print(model.summary())
evaluate_model(model, 128, 1, x_train, y_train, x_valid, y_valid)
'''

'''
Can the first layer be fully connected? def add_layer(current, random):
We don't use test data at all? def trainModel(layers):
randArchiveUpdate() threw an error, I just commented it out
'''

#TODO Erich - state to index
#TODO baian - randArchiveUpdate bug fix
#TODO Veronica - add more layer types
#TODO Erich and Veronica - update rep size
#TODO fix done function to handle 2nd termination type

