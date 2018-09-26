import mnist
import random
from collections import deque
import math

bufferSize = 1000
totalSetSize = 10000 #len(mnist.trainImages)
accesses = deque()
nextAccess = []
buffer = {}

for i in range(totalSetSize):
    nextAccess.append(deque())

def prepareBuffer(totalAccesses):
    for i in range(totalAccesses):
        id = random.randrange(totalSetSize)
        accesses.append(id)
        nextAccess[id].append(i)

def getTest():
    id = accesses.popleft()
    accessTime = nextAccess[id].popleft()
    if len(nextAccess[id]) > 0:
        nextAccessTime = nextAccess[id][0]
    else:
        nextAccessTime = math.inf
    
    if accessTime in buffer:
        x, y = buffer[accessTime]
        del buffer[accessTime]
    else:
        x, y = mnist.getTrainingSample(id)

    if nextAccessTime != math.inf and len(buffer) < bufferSize:
        buffer[nextAccessTime] = x, y
    elif nextAccessTime != math.inf:
        maxAccessTime = max(buffer, key=int)
        if maxAccessTime > nextAccessTime:
            del buffer[maxAccessTime]
            buffer[nextAccessTime] = x, y

    return x, y

prepareBuffer(100001)
for i in range(100001):
    getTest()
    if i % 1000 == 0:
        print(mnist.accesses, i)

