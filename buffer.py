import mnist
import random
from collections import deque
import math

bufferSize = 100
totalSetSize = 10000 #len(mnist.trainImages)
queries = 100000
accesses = deque()
nextAccess = []
buffer = {}
maxTimeInBuffer = -1

for i in range(totalSetSize):
    nextAccess.append(deque())

def prepareBuffer(totalAccesses):
    for i in range(totalAccesses):
        id = random.randrange(totalSetSize)
        accesses.append(id)
        nextAccess[id].append(i)

def getTest():
    global maxTimeInBuffer
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
        maxTimeInBuffer = max(maxTimeInBuffer, nextAccessTime)
    elif nextAccessTime != math.inf:
        if maxTimeInBuffer > nextAccessTime:
            del buffer[maxTimeInBuffer]
            buffer[nextAccessTime] = x, y
            maxTimeInBuffer = nextAccessTime

    return x, y

prepareBuffer(queries)
for i in range(queries):
    getTest()
print('Buffer contendo no máximo ', bufferSize, ' imagens ou ', 100*bufferSize/totalSetSize, \
            '% da base de dados, após ', queries, ' consultas, ', 100*mnist.accesses/queries, '% não estavam presentes no buffer.', sep='')