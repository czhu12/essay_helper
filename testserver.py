chars = [u'\n',
 u' ',
 u'"',
 u'%',
 u'&',
 u"'",
 u'(',
 u')',
 u'*',
 u'+',
 u',',
 u'-',
 u'.',
 u'/',
 u'0',
 u'1',
 u'2',
 u'3',
 u'4',
 u'5',
 u'6',
 u'7',
 u'8',
 u'9',
 u':',
 u';',
 u'=',
 u'>',
 u'?',
 u'[',
 u']',
 u'_',
 u'a',
 u'b',
 u'c',
 u'd',
 u'e',
 u'f',
 u'g',
 u'h',
 u'i',
 u'j',
 u'k',
 u'l',
 u'm',
 u'n',
 u'o',
 u'p',
 u'q',
 u'r',
 u's',
 u't',
 u'u',
 u'v',
 u'w',
 u'x',
 u'y',
 u'z',
 u'~',
 u'\xa9',
 u'\xb0',
 u'\xb4',
 u'\xb5',
 u'\xe4',
 u'\xe5',
 u'\xe6',
 u'\xef',
 u'\xf3',
 u'\xf6',
 u'\xfa',
 u'\xfc',
 u'\u03ac',
 u'\u03af',
 u'\u03b1',
 u'\u03b2',
 u'\u03b4',
 u'\u03b7',
 u'\u03b9',
 u'\u03ba',
 u'\u03bb',
 u'\u03bc',
 u'\u03bd',
 u'\u03bf',
 u'\u03c0',
 u'\u03c1',
 u'\u03c2',
 u'\u03c3',
 u'\u03c4',
 u'\u03c7',
 u'\u03c8',
 u'\u03c9',
 u'\u03cc',
 u'\u03cd',
 u'\u2013',
 u'\u2014',
 u'\u2019',
 u'\u201c',
 u'\u201d',
 u'\u202f',
 u'\u2032',
 u'\u2192',
 u'\u2212',
 u'\u2248']
maxlen = 40


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import load_model

from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

model = load_model('model.h5')

def serialize(sentence):
    if len(sentence) != maxlen:
        raise ValueError("sentence must be of length: {}".format(maxlen))

    X = np.zeros((maxlen, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentence):
        for t, char in enumerate(sentence):
            X[t, char_indices[char]] = 1

    return np.expand_dims(X, axis=0)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def predict(X, diversity=1.0):
    prediction_seq = []
    preds = model.predict(X)[0]
    next_index = sample(preds, diversity)
    count = 1
    # While we have not predicted a full word, keep predicting.
    while count <= maxlen:
        prediction_seq.append(next_index)
        X = np.roll(X, -1)
        X[0, -1, :].fill(0)
        X[0, -1, next_index] = 1
        count += 1

        # Do we need a mutex on this model
        preds = model.predict(X)[0]
        print('predicted: {}'.format(next_index))
        next_index = sample(preds, diversity)
    return prediction_seq

def deserialize(predictions):
    next_word = ''
    print(predictions)
    for i in range(len(predictions)):
        p = predictions[i]
        next_word += indices_char[p]

    return next_word

@app.route('/autocomplete', methods=["POST"])
def autocomplete():
    sentence = request.json['sentence']
    ## Takes a sentence and converts it to a (1, maxlen, len(chars)) one hot encoded sequence
    #X = serialize(sentence)
    #print(X)
    ## Takes a batch of sequences and converts it to a index encoded sequence
    ## of variable length.
    #prediction = predict(X)
    #print(prediction)
    ## Takes as input a index encoded sequence and returns the equivalent text
    #output = deserialize(prediction)
    ## Given a sentence, serialize it.


    generated = ''
    for i in range(40):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return jsonify({'next_word': generated})

if __name__ == "__main__":
    app.run()
