import os
import functools
import operator
import gzip
import struct
import array
import tempfile
import random
import numpy as np

# Interpreta um documento no formato IDX (formato das imagens do mnist)
path = 'Images/'
def parse_idx(fd):
	DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)
	
	header = fd.read(4)

	_, data_type, num_dimensions = struct.unpack('>HBB', header)
	data_type = DATA_TYPES[data_type]
	dimension_sizes = struct.unpack('>' + 'I' * num_dimensions, fd.read(4 * num_dimensions))

	data = array.array(data_type, fd.read())
	data.byteswap()

	if(len(dimension_sizes) == 3):
		return np.array(data).reshape((dimension_sizes[0], dimension_sizes[1] * dimension_sizes[2]))
	else:
		return np.array(data).reshape(dimension_sizes)

def trainingImages():
	return parse_idx(open(path + 'train-images.idx3-ubyte', 'rb'))
def trainingLabels():
	return parse_idx(open(path + 'train-labels.idx1-ubyte', 'rb'))
def testingImages():
	return parse_idx(open(path + 't10k-images.idx3-ubyte', 'rb'))
def testingLabels():
	return parse_idx(open(path + 't10k-labels.idx1-ubyte', 'rb'))

trainImages = trainingImages()
trainLabels = trainingLabels()
testImages = testingImages()
testLabels = testingLabels()
accesses = 0

# Recupera a amostra de id dado e conta um acesso direto a database
def getTrainingSample(id):
	global accesses
	accesses += 1
	return trainImages[id], trainLabels[id]
