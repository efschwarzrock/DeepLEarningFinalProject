import json

class saveArchitecture:
    def __init__(self, name):
        self.name = name
        self.architectures = {}

    # taking in a list of tuple:
    # (layerName, shape)
    # and its accuracy
    # convert it to String and add it to the dictionary
    def archiveFindings(self, list, acc):
        self.architectures[self.toString(list)] = acc

    def toString(self, list):
        result = ""
        for item in list:
            result += json.dumps(item)
            result += "|"

        return result[0:len(result)-1]

    def getValueByString(self, modelString):
        return self.architectures[modelString]

    def getValue(self, model):
        modelString = self.toString(model)
        if modelString in self.architectures.keys():
            return self.architectures[modelString]
        else:
            return None

    def getKeys(self):
        return self.architectures.keys()

