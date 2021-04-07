class saveArchitecture:
    def __init__(self, name):
        self.name = name
        self.architectures = {}

    # taking in a list of tuple:
    # (layerName, shape)
    # and its accuracy
    # convert it to String and add it to the dictionary
    def add(self, list, acc):
        self.architectures[self.toString(list)] = acc

    def toString(self, list):
        result = ""
        for item in list:
            name, shape = item
            result += name
            for i in shape:
                result += "(" + str(i) + ") "
            result += "---"
        return result

    def getValue(self, list):
        return self.architectures[self.toString(list)]


