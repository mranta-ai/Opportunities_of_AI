## Text analysis example - IMDB

In this example, we build a simple neural network model to predict the sentiment of movie reviews.

First, we load the IMDB data that is included in the **Keras** library (part of **Tensorflow**). Also, we load the **preprocessing** module.

from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing

This is a dataset of 25,000 movies reviews from IMDB, labelled by sentiment (positive/negative). The reviews have been preprocessed, and each review is encoded as a list of word indexes (integers). 

Words are ranked by how often they occur (in the training set) and only the **num_words** most frequent words are kept. Any less frequent word will appear as `oov_char` value in the sequence data. If we use **num_words = None**, all words are kept.

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)

The following commands pad sequences to the same length, in this case, to 20 words.

**pad_sequences()** creates a 2D Numpy array of shape (number of samples x number of words) from a list of sequences.

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=20)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=20)

x_train.shape

y_train.shape

### Densely connected network

We first build a traditional densely connected feed-forward-network. We also need an Embedding layer to code our words efficiently and a Flatten layer to transform our 2D-tensor to 1D-vector.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding

Our embedding layer codes 10000 words to 8-element vectors. The output layer has one neuron and a sigmoid-activation function that gives a probability for positive/negative. **model.sequential()** defines the network type, and the **add()** -functions are used to add layers to the model.

model = Sequential()
model.add(Embedding(10000, 8, input_length=20))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

Like with the examples of the computer vision section, we can stick with the **RMSprop** gradient descent optimiser. Because we are doing positive/negative classification, binary_crossentropy is the correct loss function. We measure the model performance with prediction accuracy.

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

The model has 80161 parameters.

model.summary()

The data is split into training and validation parts with 80/20% division. We go through the data ten times (**epochs=10**). The data is fed to the model in 32 unit batches and, thus, each epoch has 625 steps (32 * 625 = 20000). Our prediction accuracy with the validation data is 0.75. However, the model appears to be overfitting as the validation loss is increasing, and there is a wide gap between the training accuracy and the validation accuracy in the last epochs.

history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

import matplotlib.pyplot as plt
plt.style.use('bmh')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r--', label='Training accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r--', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.legend()
plt.show()

As our first improvement, we could try to use pre-trained embeddings in our model. Word embeddings include semantic information about our words (words appearing in similar contexts are close to each other). Pretrained embeddings are trained using vast amounts of text (billions of words). One could assume that the semantic information in these pre-trained embeddings is of higher quality and should improve our predictions. Let's see...

To be able to use this approach, we need the original IMBD data. Search for aclimdb.zip from the internet.

import os

My raw data is in the *aclImdb* -folder under the work folder

imdb_raw = './aclImdb/'

First, we define empty lists for the reviews and their sentiment labels. Then we collect the negative reviews from *./aclImdb/train/neg* -folder. We also add to the labels-list zero for these cases. A similar approach is repeated for the positive reviews. Thus, in our lists, we have first the negative reviews and the positive reviews.

labels = []
texts = []

# Collect negative reviews
train_neg_dir = os.path.join(imdb_raw,'train','neg')
for file in os.listdir(train_neg_dir):
    f = open(os.path.join(train_neg_dir, file))
    texts.append(f.read())
    f.close()
    labels.append(0)

# Collect positive reviews
train_neg_dir = os.path.join(imdb_raw,'train','pos')
for file in os.listdir(train_neg_dir):
    f = open(os.path.join(train_neg_dir, file))
    texts.append(f.read())
    f.close()
    labels.append(1)

Below is an example text and its' sentiment (0=negative).

texts[0]

labels[0]

We need Numpy and text-processing tools from the Keras libary.

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

The following commands tokenise words into vectors.

tokenizer = Tokenizer(num_words = 10000)

tokenizer.fit_on_texts(texts)

The following commands transform each text in texts to a sequence of integers.

Only words known by the tokenizer will be taken into account. It will take into account only the 10000 most frequent words.

sequences = tokenizer.texts_to_sequences(texts)

Now, we use longer texts. We keep the 200 first words from each review.

data = pad_sequences(sequences, maxlen=200)

The following command transforms the labels list to a numpy array.

labels = np.asarray(labels)

data.shape

labels.shape

Because the reviews are in order (all the negative reviews first and then the positive reviews), we have to shuffle the data before feeding it to the model.

indices = np.arange(25000)

np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]

80 / 20 % separation of the data to training and validation parts.

x_train = data[:20000]
y_train = labels[:20000]
x_val = data[20000: 25000]
y_val = labels[20000: 25000]

The Stanford NLP group offers GLOVE pre-trained embeddings. You can download them from [nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/). We use the glove6B.zip that is trained using 6 billion tokens. Each word is represented as a 100-dimensional vector.

# we use 100-dimensional vectors
embeddings_index = {}
f = open(os.path.join('./glove.6B/', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

GLOVE has 400k tokens.

len(embeddings_index)

We build the embedding matrix by going through our word index and adding its' embeddings from the Glove model (if it is found).

embedding_matrix = np.zeros((10000, 100))
for word, i in word_index.items():
    if i < 10000:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

Because our model uses now 100-dimensional word vectors, the network also has a lot of more parameters. Our network also has a new 32-neuron dense layer after the Flatten-layer.

model = Sequential()
model.add(Embedding(10000, 100, input_length=200))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

We set the weights of the embedding layer using the Glove weights in the embedding matrix. The weights need to be locked so that we are not retraining them with our small dataset.

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

Again, we use the RMSprop optimiser, the binary_crossentropy loss function and accuracy as our performance metric.

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history = model.fit(x_train, y_train,epochs=10,batch_size=32,validation_data=(x_val, y_val))

Not a good performance. Heavy overfitting and worse accuracy. Let's try something else.

plt.style.use('bmh')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r--', label='Training acc')
plt.plot(epochs, val_acc, 'b--', label='Validation acc')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r--', label='Training loss')
plt.plot(epochs, val_loss, 'b--', label='Validation loss')
plt.legend()
plt.show()

### Recurrent neural networks

Next thing that we can try is to use Recurrent neural networks. They are especially efficient for sequences like texts.

![RNN](./images/rnn.svg)

from tensorflow.keras.layers import SimpleRNN

Now, instead of a Flatten() layer, we have a SimpleRNN() layer.

model = Sequential()
model.add(Embedding(10000, 100, input_length=200))
model.add(SimpleRNN(100))
model.add(Dense(1, activation='sigmoid'))
model.summary()

Again, we use the GLOVE weights.

# Load GLove wieghts
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.summary()

Nothing has changed in the compile() and fit() -steps.

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history = model.fit(x_train, y_train,epochs=10,batch_size=32,validation_data=(x_val, y_val))

Well, overfitting is not such a serious problem any more, but the performance is not improving still.

plt.style.use('bmh')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r--', label='Training accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r--', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.legend()
plt.show()

### Long short-term memory

As our last idea, we try the LSTM-version of RNN. It has achieved very good performance in practice, so, let's hope for the best.
![lstm](./images/lstm.svg)

from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(Embedding(10000, 100, input_length=200))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Load GLove wieghts
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.summary()

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history = model.fit(x_train, y_train,epochs=10,batch_size=32,validation_data=(x_val, y_val))

Finally, we see some progress! Now the accuracy is around 87 %. So, a very significant improvement in performance. For the exact evaluation of performance, we should use a separate test set. However, the validation dataset accuracy gives a good indication of the performance of our model.

plt.style.use('bmh')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r--', label='Training accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r--', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.legend()
plt.show()

As our final model, let's test what kind of effect the predetermined weights have for the performance and train an LSTM model from scratch.

model = Sequential()
model.add(Embedding(10000, 32, input_length=200))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history = model.fit(x_train, y_train,epochs=10,batch_size=32,validation_data=(x_val, y_val))

Because there are no locked parameters, the number of trainable parameters increases, and this causes some overfitting. However, the performance is at the same level as in the previous model. So, the predetermined weights do not appear to improve the accuracy, but they help at fighting overfitting.

plt.style.use('bmh')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r--', label='Training accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r--', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.legend()
plt.show()

