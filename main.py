from handle_data import loadDataset
from neighbors import getNeighbors
from response import getResponse
from accuracy import getAccuracy


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('dataset/iris.data', split, trainingSet, testSet)
    print('Train Set : ' + repr(len(trainingSet)))
    print('Test Set : ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) +
              ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('accuracy:' + repr(accuracy) + '%')


main()
