import numpy as np

MAXLAYERS = 5

STATENAMES = {"Convolution", "Pooling", "FullyConected", "Termination"}

def getConvolutionStateIndex(state):
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

    return 0


def getPoolingStateIndex(state):
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

    return 1


def getFullStateIndex(state):
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
    if state[0] == STATENAMES[0]:
        return getConvolutionStateIndex(state)
    elif state[0] == STATENAMES[1]:
        return getPoolingStateIndex(state)
    elif state[0] == STATENAMES[2]:
        return getFullStateIndex(state)
    elif state[0] == STATENAMES[3]:
        return getTerminationStateIndex(state)

NUMACTIONS = 4
NumTerminationActions = 1
NUMSTATES = NUMACTIONS - NumTerminationActions
StateActionPairs = np.zeros(shape=(MAXLAYERS, NUMSTATES, NUMACTIONS))

print(StateActionPairs)

