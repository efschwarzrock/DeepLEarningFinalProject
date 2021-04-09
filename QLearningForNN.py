import numpy as np

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

#the state names as used in the model
STATENAMES = {"Convolution", "Pooling", "FullyConected", "Termination"}



#TODO update these function to fit new model
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

#TODO takes in a state or action and returns the index of it in the table
def getStateIndex(state):
    if state[0] == STATENAMES[0]:
        return getConvolutionStateIndex(state)
    elif state[0] == STATENAMES[1]:
        return getPoolingStateIndex(state)
    elif state[0] == STATENAMES[2]:
        return getFullStateIndex(state)
    elif state[0] == STATENAMES[3]:
        return getTerminationStateIndex(state)

#TODO takes in an index and returns the state or action it coresponds to
def getIndexState(state):
    pass

'''###########################################################################
##############################################################################
###########################################################################'''



def makeMove(action):
    #TODO takes in an action and updates the curent state and the model.
    #The current state should just have the layerdepth and what the curent layer is i.e. convo pooling ect
    #the model should be a list of the previous states so that we can build the network when the time comes
    #we may also want to store the input size so that we can update the representation size correctly
    pass





def pickBestAction(state):
    #TODO return the best action based on the state action table
    #this is the action that has the highest num value in the table
    pass


def getRandAction(state):
    #TODO return a random action that state can move to
    pass


def pickMove(state):
    r = np.random.rand()
    if r > epsilon:
        return pickBestAction(state)
    else:
        return getRandAction(state)

def update(oldState, move, newState, accuracy):
    #TODO update to change correct things
    old = stateActionPairs[oldState[0]][oldState[1]][move]
    return old*(1-stepSize) + stepSize*(accuracy + discout*(np.max(stateActionPairs[newState[0]][newState[1]])))

def done():
    #TODO return true if we reached a termination state
    pass


def trainModel():
    #TODO see if we allready built that model and if not build and train the model, return the accuracy
    #Model will most likly be a global variable
    pass

def getOriginalState():
    #TODO return the original state
    pass

def archiveFindings(accuracy):
    #TODO store the model and the resulting accuracy in the archive
    pass

def randArchiveUpdate():
    #TODO randomly sample 100 models from the dictionary and update the Q Values(IDK what exactly its supposed to do)
    #its described on page 6 paragraph right under the table
    pass

def epsilonDecay(gens):
    #TODO reduce epsilon so the model becomes more and more greedy
    pass


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
curState = getOriginalState()

StateActionPairs = np.zeros(shape=(MAXLAYERS, NUMSTATES, NUMACTIONS))

#the number of generations/trials run(howmany models have we trained)
gens = 0

#a debugging incremetor
its = 0





while gens < 10000:
    move = pickMove(curState)
    oldState = list(curState)#TODO update to fit archetecture
    makeMove(move)
    newState = list(curState)#TODO update to fit archetecture

    if done():
        accuracy = trainModel()
        archiveFindings(accuracy)
        update(oldState, move, newState, accuracy)
        curState = getOriginalState()
        gens = gens + 1
        randArchiveUpdate()
        epsilonDecay(gens)
    else:
        update(oldState, move, newState, 0)


    its = its+1















