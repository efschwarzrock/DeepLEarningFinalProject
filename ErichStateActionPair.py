import numpy as np
import random

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
def add_layer(current):
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
    rand = random.randint(0, len(next_layers) - 1)
    next_layer = next_layers[rand]

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


# generate a random architecture with specified parameters
def generate_architecture():
    layers = [add_layer(0)]
    while (layers[-1]['type'] != 'softmax'):
        layers.append(add_layer(layers[-1]))
    return layers


MAXLAYERS = 5

STATENAMES = {"Convolution", "Pooling", "FullyConected", "Termination"}

def getConvolutionStateIndex(state):
    RecepFieldSize = state['filter_size'] #{1, 3, 5}
    #the index of the size
    RecepFieldSizeI = 0


    NumRecepFields = state['num_filters'] #{64, 128, 256, 512}
    #index of the numRecepFields
    NumRecepFieldsI = 0


    RepSize = state['representation_size'] #{(∞, 8], (8, 4], (4, 1]}
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

    return 0


def getPoolingStateIndex(state):
    RecepFieldSize = state['pool_size']  # {(5,(3,(2}
    # the index of the size
    RecepFieldSizeI = 0


    RepSize = state['representation_size']  # {(∞, 8], (8, 4], (4, 1]}
    # index of RepSize
    RepSizeI = 0

    if RecepFieldSize == 2:
        RecepFieldSizeI = 0
    elif RecepFieldSize == 3:
        RecepFieldSizeI = 1
    elif RecepFieldSize == 5:
        RecepFieldSizeI = 2

    ret = 3*3*4 + RecepFieldSizeI*3 + RepSizeI

    return 1


def getFullStateIndex(state):
    #the number of fully conected layers befor this one
    NumConsecFullConec = state['consecutive'] # 0, 1

    NumNeurons = state['nodes'] # {512, 256, 128}
    NumNeuronsI = 0

    if NumNeurons == 512:
        NumNeuronsI = 0
    elif NumNeurons == 256:
        NumNeuronsI = 1
    elif NumNeurons == 128:
            NumNeuronsI = 2
    ret = 3*3*4 + 3*3 + NumConsecFullConec*3 + NumNeuronsI
    return 2


def getTerminationStateIndex(state):
    TermType = state[1]
    TypeI = 0
    if TermType == "Softmax":
        TypeI = 0
    else:
        TypeI = 1


    ret = 3 * 3 * 4 + 3 * 3 + 3 * 2 + TypeI
    return 3


def getStateIndex(state):
    if state['type'] == STATENAMES[0]:
        return getConvolutionStateIndex(state)
    elif state['type'] == STATENAMES[1]:
        return getPoolingStateIndex(state)
    elif state['type'] == STATENAMES[2]:
        return getFullStateIndex(state)
    elif state['type'] == STATENAMES[3]:
        return getTerminationStateIndex(state)

NUMACTIONS = 4
NumTerminationActions = 1
NUMSTATES = NUMACTIONS - NumTerminationActions
StateActionPairs = np.zeros(shape=(MAXLAYERS, NUMSTATES, NUMACTIONS))

architecture = generate_architecture()
for a in architecture:
    print(a)
    print(a['type'])

