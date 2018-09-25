import mnist
import random

bufferSize = 10000
accesses = []
nextAccess = []
bufferIds = []
bufferImages = []
bufferLabels = []

for i in range(len(mnist.trainImages)):
    nextAccess.append([])

def prepareBuffer(totalAccesses):
    for i in range(totalAccesses):
        id = random.randrange(len(mnist.trainImages))
        accesses.append(id)
        nextAccess[id].append(i)

def getTest():
    id = accesses.pop(0)
    nextAccess[id].pop(0)

    LatestToUtilize = 0
    for i in range(len(bufferIds)):
        if id == bufferIds[i]:
            return bufferImages[i], bufferLabels[i]
        elif len(nextAccess[bufferIds[LatestToUtilize]]) == 0:
            continue
        elif len(nextAccess[bufferIds[i]]) == 0 or nextAccess[bufferIds[LatestToUtilize]][0] < nextAccess[bufferIds[i]][0]:
            LatestToUtilize = i

    img, lbl = mnist.getTrainingSample(id)
    if(len(bufferIds) < bufferSize):
        bufferIds.append(id)
        bufferImages.append(img)
        bufferLabels.append(lbl)
    else:
        bufferIds[LatestToUtilize] = id
        bufferImages[LatestToUtilize] = img
        bufferLabels[LatestToUtilize] = lbl
    return img, lbl

prepareBuffer(100001)
for i in range(100001):
    getTest()
    if i % 1000 == 0:
        print(mnist.accesses, i)

