import tensorflow as tf

import mnistbuffer
import mnist

# Preparar buffer
queries = 1000
batchSize = 100
mnistbuffer.prepare(queries * batchSize)

# Criar modelo de rede neural
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Definir função de perda e otimizador
y_ = tf.placeholder(tf.int64, [None])
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Treinar rede, recuperando exemplos através do buffer
for i in range(queries):
    if(i % 100 == 0):
        print("Iteração de treino: " + str(i))
    batch_xs, batch_ys = mnistbuffer.getBatch(batchSize)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Testar modelo treinado
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accuracy = sess.run(
    accuracy, feed_dict={
        x: mnist.testImages,
        y_: mnist.testLabels
    })

print()
print("Fim do treinamento")
print("Precisão da rede neural: " + str(accuracy))
print()
print("Buscas: " + str(queries * batchSize))
print("Colisões no buffer: " + str((queries * batchSize) - mnist.accesses))
print("Frequencia de colisão no buffer: " + str(((queries * batchSize) - mnist.accesses) / (queries * batchSize)))
print()
print("Tamanho do buffer: " + str(mnistbuffer.bufferSize))
print("Tamanho da database: " + str(mnistbuffer.totalSetSize))
print("Tamanho do buffer / Tamanho da database: " + str(mnistbuffer.bufferSize / mnistbuffer.totalSetSize))
