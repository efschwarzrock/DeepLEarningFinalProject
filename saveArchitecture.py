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

            # if name is "convolution":
            #     nfilters = item["num_filters"]
            #     filter_size = item["filter_size"]
            #     stride = item["stride"]
            #     representation_size = item["representation_size"]
            #     layer_depth = item["layer_depth"]
            #     result += str(nfilters) \
            #             + "," + str(filter_size) \
            #             + "," + str(stride) \
            #             + "," + str(representation_size) \
            #             + "," + str(layer_depth)
            #
            # elif name is "pooling":
            #     pool_size = item[1]
            #     stride = item[2]
            #     layer_depth = item[3]
            #     result += str(pool_size) \
            #             + "," + str(stride) \
            #             + "," + str(layer_depth)
            #
            # elif name is "dense":
            #     nnodes = item[1]
            #     layer_depth = item[2]
            #     consecutive_FC = item[3]
            #     result += str(nnodes) \
            #             + "," + str(layer_depth) \
            #             + "," + str(consecutive_FC)
            #
            # elif name is "softmax":
            #     nnodes = item[1]
            #     layer_depth = item[2]
            #     result += str(nnodes) \
            #             + "," + str(layer_depth)

    def getValueByString(self, list):
        return self.architectures[list]

    def getKeys(self):
        return self.architectures.keys()

