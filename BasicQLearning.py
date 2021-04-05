

import numpy as np

gridworld = [[0, 0, 0, 50, 0, 0, 0, 0],
             [0, 0, 0,-2, 0, 0, 0, 0],
             [0, 0, 0,-2, 0, 0, 0, 0],
             [-2,0, 0,-2, 0,-2, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0,-2, 3, 0],
             [0, 0, 0, 0, 0, 0, 0, -2]]



actualMovePercent = .8
epsilon = .8
originalState = [6, 0]
stepSize = .1
discout = .9


curState = list(originalState)
halfPercent = (1-actualMovePercent)/2

def actualyMove(direction):
    if direction == 0 and curState[0] < len(gridworld)-1:
        curState[0] = curState[0]+1
    elif direction == 1 and curState[1] < len(gridworld[curState[0]])-1:
        curState[1] = curState[1] + 1
    elif direction == 2 and curState[0] > 0:
        curState[0] = curState[0]-1
    elif direction == 3 and curState[1] > 0:
        curState[1] = curState[1] - 1



def makeMove(direction):
    r = np.random.rand()
    actdirection = direction
    if r<halfPercent:
        actdirection = direction-1%4
    elif r<2*halfPercent:
        actdirection = direction+1%4
    actualyMove(actdirection)


def printGridWorld():
    for a in range(0, len(gridworld)):
        line = ""
        for b in range(0, len(gridworld[a])):
            if [a, b] == curState:
                line = line + "s  ,"
            else:
                s = str(gridworld[a][b])
                s = s.ljust(3)
                line = line + s + ','
        print(line)


def pickBestMove(state):
    options = stateActionPairs[state[0]][state[1]]
    indexs = np.argwhere(options == np.amax(options)).flatten()
    x = indexs[np.random.randint(0, len(indexs))]
    if state == [0, 7]:
        stuckArr[len(indexs)-1][x] = stuckArr[len(indexs)-1][x] + 1
    return x

def pickMove(state):
    r = np.random.rand()
    if r < epsilon:
        return pickBestMove(state)
    else:
        x = np.random.randint(0, 4)
        if state == [0, 7]:
            stuckArr[3][x] = stuckArr[3][x] + 1
        return x

def update(oldState, move, newState):
    old = stateActionPairs[oldState[0]][oldState[1]][move]
    return old*(1-stepSize) + stepSize*(gridworld[newState[0]][newState[1]] + discout*(np.max(stateActionPairs[newState[0]][newState[1]])))

def done():
    if gridworld[curState[0]][curState[1]]:
        return True
    else:
        return False


def getDirection(actions):
    index = np.argmax(actions)
    if index == 0:
        return " V "
    elif index == 1:
        return " > "
    elif index == 2:
        return " ^ "
    elif index == 3:
        return " < "


def printSimpleLearned():
    for i in range(0, len(gridworld)):
        line = ""
        for j in range(0, len(gridworld[i])):
            if gridworld[i][j] == 0:
                line = line + getDirection(stateActionPairs[i][j]) + ","
            else:
                s = str(gridworld[i][j])
                s = s.ljust(3)
                line = line + s + ','
        print(line)

def numToStrLength(num, length):
    if num > 99:
        num = 99
    if num < -99:
        num = -99
    s = str(num)
    s = s.center(length)
    return s


def printSingle(i, param):
    line = ""
    for j in range(0, len(gridworld[i])):
        line = line + "|" + numToStrLength(stateActionPairs[i][j][param], 12) + "|"
    print(line)

def printSingleDir(i, param):
    line = ""
    for j in range(0, len(gridworld[i])):
        if stateActionPairs[i][j][param] == 0:
            line = line + "|     |"
        elif stateActionPairs[i][j][param] > 0:
            if param == 2:
                line = line + "|  ^  |"
            else:
                line = line + "|  v  |"
        else:
            line = line + "|  X  |"
    print(line)


def printMiddle(i):
    line = ""
    for j in range(0, len(gridworld[i])):
        line = line + "|" + numToStrLength(stateActionPairs[i][j][3], 4) + numToStrLength(gridworld[i][j], 4) + numToStrLength(stateActionPairs[i][j][1], 4) + "|"
    print(line)

def printMiddleDir(i):
    line = ""
    for j in range(0, len(gridworld[i])):
        if stateActionPairs[i][j][3] == 0:
            line = line + "| "
        elif stateActionPairs[i][j][3] > 0:
            line = line + "|<"
        else:
            line = line + "|X"

        line = line + numToStrLength(gridworld[i][j], 3)

        if stateActionPairs[i][j][1] == 0:
            line = line + " |"
        elif stateActionPairs[i][j][1] > 0:
            line = line + ">|"
        else:
            line = line + "X|"
    print(line)

def printRow(i):
    line = ""
    print(line.ljust(len(gridworld[i])*14, "_"))
    printSingle(i, 2)
    printMiddle(i)
    printSingle(i, 0)
    line = ""
    print(line.ljust(len(gridworld[i]) * 14, "_"))

def printRowDir(i):
    line = ""
    print(line.ljust(len(gridworld[i])*7, "_"))
    printSingleDir(i, 2)
    printMiddleDir(i)
    printSingleDir(i, 0)
    line = ""
    print(line.ljust(len(gridworld[i]) *7, "_"))




def printComplexLearnedVals():
    for i in range(0, len(gridworld)):
            printRow(i)

def printComplexLearnedDir():
    for i in range(0, len(gridworld)):
        printRowDir(i)

def printArr(numVisits):
    for i in range(0, len(numVisits)):
        line = ""
        for j in range(0, len(numVisits[i])):
            s = str(numVisits[i][j])
            line = line + s.center(6) + ","
        print(line)


printGridWorld()
#           num taken
stuckArr = [[0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]]
#
stateActionPairs = np.zeros(shape=(len(gridworld), len(gridworld[0]), 4))
numVisits = np.zeros(shape=(len(gridworld), len(gridworld[0])))
gens = 0
its = 0
while gens < 10000:
    numVisits[curState[0]][curState[1]] = numVisits[curState[0]][curState[1]] + 1
    move = pickMove(curState)
    oldState = list(curState)
    makeMove(move)
    newState = list(curState)
    stateActionPairs[oldState[0]][oldState[1]][move] = update(oldState, move, newState)

    if done():
        curState = list(originalState)
        gens = gens + 1
    its = its+1
    if its%1000 == 0:
        its = 0
        epsilon = epsilon - .1
        #print(moves)
        #print(curState)
        #printGridWorld()


print("\n\n")
printSimpleLearned()
print("\n\n")


stateActionPairs = np.around(stateActionPairs, decimals=1)




printComplexLearnedVals()
print("\n\n\n")
#printComplexLearnedDir()





printArr(numVisits)

print(stuckArr)

'''  2
    3s1
     0
'''