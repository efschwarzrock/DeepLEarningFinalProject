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
            result += "|"
            name = item[0]

            if name is "convolution":
                nfilters = item[1]
                filter_size = item[2]
                stride = item[3]
                representation_size = item[4]
                layer_depth = item[5]
                result += str(nfilters) \
                        + "," + str(filter_size) \
                        + "," + str(stride) \
                        + "," + str(representation_size) \
                        + "," + str(layer_depth)

            elif name is "pooling":
                pool_size = item[1]
                stride = item[2]
                layer_depth = item[3]
                result += str(pool_size) \
                        + "," + str(stride) \
                        + "," + str(layer_depth)

            elif name is "dense":
                nnodes = item[1]
                layer_depth = item[2]
                consecutive_FC = item[3]
                result += str(nnodes) \
                        + "," + str(layer_depth) \
                        + "," + str(consecutive_FC)

            elif name is "softmax":
                nnodes = item[1]
                layer_depth = item[2]
                result += str(nnodes) \
                        + "," + str(layer_depth)

        return result

    def getValue(self, list):
        return self.architectures[self.toString(list)]


