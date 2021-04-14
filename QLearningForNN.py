import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import random as r
import copy

#Chance of taking a random action
epsilon = .8

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
saved_architecture = {}

current_layer = 0


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
    # get possible layers and current layer depth
    layers = get_layers()
    current_depth = 0
    if current != 0:
        current_depth = current['layer_depth']

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
        next_layer = find_best(current, next_layers, current_depth)

    # update layer depth
    next_layer['layer_depth'] = current_depth + 1

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
    best = 0
    best_layer = None
    for layer in next_layers:
        current_index = getStateIndex(current)
        next_index = getStateIndex(layer)
        accuracy = StateActionPairs[depth][current_index][next_index]
        if accuracy > best:
            best = accuracy
            best_layer = layer
    return layer

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
    layers.append(action)
    current_layer = action
    pass

#return the best next layer based on the state action table
def getBestNextLayer(current_layer):
    return add_layer(current_layer, False)

#return a random next layer
def getRandNextLayer(current_layer):
    return add_layer(current_layer, True)

#generate next layer
def getNextLayer(current_layer):
    r = np.random.rand()
    if r > epsilon:
        return getBestNextLayer(current_layer)
    else:
        return getRandNextLayer(current_layer)
    
def update(oldState, newState, accuracy):
    old = StateActionPairs[oldState['layer_depth']][getStateIndex(oldState)][getStateIndex(newState)]
    return old*(1-stepSize) + stepSize*(accuracy + discout*(np.max(StateActionPairs[newState['layer_depth']][getIndexState(newState)])))

#return True if we are at a termination layer
#only have softmax for now
def done(layer):
    if layer['type'] == 'softmax':
        return True
    return False

#train architecture if we have not already, or get its reward from the table
def trainModel(layers):
    #if we've already trained this model, get from table
    if str(layers) in saved_architecture:
        accuracy = saved_architecture[str(layers)]
    #otherwise, train the model
    else:
        model = create_model(layers)
        accuracy = evaluate_model(model)
    return accuracy

#get the first layer in the architecture
def getFirstLayer():
    r = np.random.rand()
    if r > epsilon:
        return add_layer(0, False);
    else:
        return add_layer(0, True);

# take the architecture_name and accuracy, and store it
## str() it when inputting
def archiveFindings(architecture_name, accuracy):
    saved_architecture[architecture_name] = accuracy


def randArchiveUpdate():
    #TODO randomly sample 100 models from the dictionary and update the Q Values(IDK what exactly its supposed to do)
    #its described on page 6 paragraph right under the table
    #select 100 models here for now
    return r.sample(list(saved_architecture.items()), 100)

# get current epsilon given generation
def epsilonDecay(gens):
    return 1 - gens * 0.0001

def numToStrLength(num, length):
    if num > 99:
        num = 99
    if num < -99:
        num = -99
    s = str(num)
    s = s.center(length)
    return s


'''###########################################################################
##############################################################################
###########################################################################'''
#the curent state of the model
current_layer = getFirstLayer()
layers = [current_layer]

#The state action pairs
StateActionPairs = np.zeros(shape=(MAXLAYERS+1, NUMSTATES, NUMACTIONS))

#the number of generations/trials run(howmany models have we trained)
gens = 0

#a debugging incremetor
its = 0

#The input size of the next layer
inputSize = 128

x = getFirstLayer()
print(x)
print(type(x))


while gens < 10000:

    #pick a next layer
    next_layer = getNextLayer(current_layer)

    #store the old state
    oldState = copy.deepcopy(current_layer)

    #make the move and get the new state
    makeMove(next_layer)

    newState = copy.deepcopy(current_layer)


    #Check if we need to train the model
    if done(next_layer):
        #we need to train so train and archive the model
        accuracy = trainModel(layers)
        archiveFindings(str(layers), accuracy)

        #update the Q Values and give a reward of accuracy
        update(current_layer, newState, accuracy);

        #set back to original state and increase generation
        #TODO update Q value
        current_layer = getFirstLayer()
        layers = [current_layer]
        gens = gens + 1

        #Do stuff the paper says
        randArchiveUpdate()
        epsilonDecay(gens)
    else:
        #we don't need to so update the QValues with an accuracy of 0 becasue we got no reward since we aren't done
        #update(oldState, move, newState, 0)
        update(current_layer, next_layer, newState, 0)


    its = its+1




'''
# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
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

# generate random architecture and evaluate
architecture = generate_architecture()
model = create_model(architecture)
print(model.summary())
evaluate_model(model, 128, 1, x_train, y_train, x_valid, y_valid)
'''