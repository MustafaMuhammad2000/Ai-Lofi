#! /home/yiihong/ai-lofi/bin/python
""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
import matplotlib.pyplot as plt
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.preprocessing.text import one_hot
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.cluster import KMeans

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    #data = []
    notes = []
    offset = []
    duration = []

    for file in glob.glob("../donger james training data/test/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except E: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes


        for element in notes_to_parse:
            if isinstance(element, note.Note):
                #data.append('~'.join([str(element.pitch), str(element.offset), str(element.duration.quarterLength)]))
                 notes.append('n:'+str(element.pitch))
                 offset.append('o:'+str(element.offset))
                 duration.append('d:'+str(element.duration.quarterLength))
            elif isinstance(element, chord.Chord):
                #data.append('~'.join(['.'.join(str(n) for n in element.normalOrder), str(element.offset), str(element.duration.quarterLength)]))
                #notes.append('.'.join(str(n) for n in element.normalOrder))
                for i in element:
                    #print("chord",i.pitch)
                    notes.append('n:'+str(i.pitch))
                    offset.append('o:'+str(element.offset))
                    duration.append('d:'+str(element.duration.quarterLength))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    
    with open('data/offset', 'wb') as filepath:
        pickle.dump(offset, filepath)
    
    with open('data/duration', 'wb') as filepath:
         pickle.dump(duration, filepath)

    #with open('data/data', 'wb') as filepath:
    #    pickle.dump(data, filepath)

    return {'notes':notes, 'offset':offset, 'duration':duration}
    # model: [note, offset, duration] -> [x, y, z]
def generate_vocab(data):
    notes_vocab = sorted(set(i for i in data['notes']))
    notes_vocab = dict((note, number) for number, note in enumerate(notes_vocab))

    offset_vocab = sorted(set(i for i in data['offset']))
    offset_vocab = dict((note, number) for number, note in enumerate(offset_vocab))
    print('offset vocab', len(offset_vocab))
    #offset_vocab = dict((note, number+len(notes_vocab)) for number, note in enumerate(offset_vocab))

    duration_vocab = sorted(set(i for i in data['duration']))
    duration_vocab = dict((note, number) for number, note in enumerate(duration_vocab))
    #duration_vocab = dict((note, number+len(notes_vocab)+len(offset_vocab)) for number, note in enumerate(duration_vocab))

    return {'notes': notes_vocab, 'offset': offset_vocab, 'duration': duration_vocab}

if __name__ == '__main__':
    data = get_notes()
    vocab = generate_vocab(data)
    print(vocab)
    str_to_vocab_int = []
    for i, d in enumerate(data['notes']):
        str_to_vocab_int.append((vocab['notes'][d], vocab['offset'][data['offset'][i]], vocab['duration'][data['duration'][i]]))
    print(str_to_vocab_int)

    kmeans = KMeans(random_state=0, n_clusters=256).fit(str_to_vocab_int)
    y_hat = kmeans.predict(str_to_vocab_int)
    for i in y_hat:
        print(i, end=',')
    print()
    #print("y hat",y_hat, len(y_hat))

    clusters = numpy.unique(y_hat)
    #for i in clusters:
    #    row_ix = numpy.where(y_hat == i)
    #    print(row_ix)
    #    #plt.scatter(str_to_vocab_int[row_ix,0], str_to_vocab_int[row_ix, 1], str_to_vocab_int[row_ix, 2])
    print(kmeans.cluster_centers_)
    print(kmeans.labels_)
    print(kmeans.n_iter_)
    print(len(data['notes']), len(str_to_vocab_int))
    print(kmeans.transform(str_to_vocab_int).shape)
    print(kmeans.transform(str_to_vocab_int))
    #plt(kmeans.cluster_centers_)
    plt.show()