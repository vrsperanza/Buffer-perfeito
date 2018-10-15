import mnist
import random
from collections import deque
import math

bufferSize = 600
totalSetSize = len(mnist.trainImages)
queries = 100000
accesses = deque()
buffer = {}
maxTimeInBuffer = -1

# Prepara o buffer para retornar até totalAccesses acessos aleatórios na database, maximisando o numero de colisões dentro dele
def prepare(totalAccesses):
    global nextAccess
    nextAccess = []
    for i in range(totalSetSize):
        nextAccess.append(deque())
    
    for i in range(totalAccesses):
        id = random.randrange(totalSetSize)
        accesses.append(id)
        nextAccess[id].append(i)

# Retorna o próximo par (imagem, tag) da database,
# recuperando-o do buffer se estiver presente e atualizando o buffer para o próximo acesso
def getTest():
    global maxTimeInBuffer, nextAccess

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

# Retorna um grupo de size pares (imagem, tag) da database
def getBatch(size):
    batchX, batchY = [], []
    for _ in range(size):
        x, y = getTest()
        batchX.append(x)
        batchY.append(y)
    return batchX, batchY