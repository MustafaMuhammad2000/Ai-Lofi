""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from fractions import Fraction
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

def train_network():
    """ Train a Neural Network to generate music """
    # data = get_notes()['data']

    # data, songlengths = get_notes()
    # notes, offset, duration, songlengths = get_notes_multiple()

    data, songlengths = get_notes_single()
    # get amount of pitch names
    data_vocab = len(set(data))
    print(data_vocab)
    # n_vocab = len(set(notes))
    # print(n_vocab)
    # o_vocab = len(set(offset))
    # print(o_vocab)
    # d_vocab = len(set(duration))
    # print(d_vocab)


    network_input, network_output = prepare_sequences(data, data_vocab, songlengths)
    # network_input_notes, network_output_notes = prepare_sequences(notes, n_vocab, songlengths)
    # n_model = create_network(network_input_notes, n_vocab)
    # train(n_model, network_input_notes, network_output_notes, 5000, "Notes", 1024)
    #
    # network_input_offset, network_output_offset = prepare_sequences(offset, o_vocab, songlengths)
    # o_model = create_network(network_input_offset, o_vocab)
    # train(o_model, network_input_offset, network_output_offset, 100, "Offset", 512)
    #
    # network_input_duration, network_output_duration = prepare_sequences(duration, d_vocab, songlengths)
    # d_model = create_network(network_input_duration, d_vocab)
    # train(d_model, network_input_duration, network_output_duration, 200, "Duration", 512)
    #

    # model = create_network(network_input, data_vocab)
    # n_model = create_network(network_input_notes, n_vocab)
    # o_model = create_network(network_input_offset, o_vocab)
    # d_model = create_network(network_input_duration, d_vocab)
    #

    # train(model, network_input, network_output, 5000, "Data")
    # train(n_model, network_input_notes, network_output_notes, 150, "Notes")
    # train(o_model, network_input_offset, network_output_offset, 100, "Offset")
    # train(d_model, network_input_duration, network_output_duration, 125, "Duration")


def get_notes_multiple():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    songlengths = []
    totalsongs = 0
    zeros = 0

    notes = []
    offset = []
    duration = []

    for file in glob.glob("Yuger_Data/*.mid"):
        midi = converter.parse(file)
        totalsongs = totalsongs+1
        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            choice = 0
            if (len(s2.parts) > 1):
                for i in range(0, len(s2.parts)):
                    if (len(s2.parts[choice]) < len(s2.parts[i])):
                        choice = i
            print(s2.parts[choice], len(s2.parts), len(s2.parts[choice]))
            notes_to_parse = s2.parts[choice].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes


        prevOffset = 0
        length = 0
        for element in notes_to_parse:
            # print(element.offset, prevOffset)
            # currOffset = element.offset - prevOffset
            currOffset = float(Fraction(element.offset - prevOffset))
            currOffset = round(currOffset, 5)
            if(currOffset > 5):
                currOffset = 5
            # print(currOffset)
            if isinstance(element, note.Note):
                length = length+1
                notes.append(str(element.pitch))
                offset.append(str(currOffset))
                duration.append(str(element.duration.quarterLength))
                # duration.append(str(float(Fraction(element.duration.quarterLength))))
            elif isinstance(element, chord.Chord):
                length = length+1
                notes.append('.'.join(str(n) for n in element.normalOrder))
                offset.append(str(currOffset))
                duration.append(str(element.duration.quarterLength))
                # duration.append(str(float(Fraction(element.duration.quarterLength))))
            prevOffset = element.offset

        if length == 0:
            zeros = zeros + 1
            print(file)
        songlengths.append(length)

    print("This is how many files suck ass", zeros)
    print("This is length of number of songs", len(songlengths))
    print("This is the number of files", totalsongs)

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    with open('data/offset', 'wb') as filepath:
        pickle.dump(offset, filepath)

    with open('data/duration', 'wb') as filepath:
        pickle.dump(duration, filepath)

    with open('data/songlengths', 'wb') as filepath:
        pickle.dump(songlengths, filepath)
    return notes, offset, duration, songlengths

def get_notes_single():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """

    data = []
    songlengths = []
    totalsongs = 0
    zeros = 0

    for file in glob.glob("Yuger_Data/*.mid"):
        midi = converter.parse(file)
        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            choice = 0
            if (len(s2.parts) > 1):
                for i in range(0, len(s2.parts)):
                    if (len(s2.parts[choice]) < len(s2.parts[i])):
                        choice = i
            print(s2.parts[choice], len(s2.parts), len(s2.parts[choice]))
            notes_to_parse = s2.parts[choice].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        prevOffset = 0
        length = 0
        for element in notes_to_parse:
            currOffset = element.offset-prevOffset
            currOffset = float(Fraction(element.offset - prevOffset))
            currOffset = round(currOffset, 5)
            if(currOffset > 5):
                currOffset = 5
            # print(currOffset)
            if isinstance(element, note.Note):
                data.append((str(element.pitch), str(currOffset), str(element.duration.quarterLength)))
                length = length+1
            elif isinstance(element, chord.Chord):
                data.append(('.'.join(str(n) for n in element.normalOrder), str(currOffset), str(element.duration.quarterLength)))
                length = length+1
            prevOffset = element.offset
        if length == 0:
            zeros = zeros + 1
            print(file)
        songlengths.append(length)

    print("This is how many files suck ass", zeros)
    print("This is length of number of songs", len(songlengths))
    print("This is the number of files", totalsongs)

    with open('data/data', 'wb') as filepath:
        pickle.dump(data, filepath)
    with open('data/songlengths', 'wb') as filepath:
        pickle.dump(songlengths, filepath)

    return data, songlengths


def prepare_sequences(notes, n_vocab, songlengths):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 32

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))
    print(pitchnames)
    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    print(note_to_int)
    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    # for i in range(0, len(notes) - sequence_length, 1):
    #     sequence_in = notes[i:i + sequence_length]
    #     sequence_out = notes[i + sequence_length]
    #     network_input.append([note_to_int[char] for char in sequence_in])
    #     network_output.append(note_to_int[sequence_out])

    j = 0
    i = 0
    total = songlengths[j]
    while i < len(notes)-sequence_length:
        if(sequence_length+i <= total):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            # print([note_to_int[i] for i in sequence_in])
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
            i = i+1
        else:
            i = total
            j = j+1
            total = total+songlengths[j]
    print(j)


    print("This is network input", len(network_input), "This is network output", len(network_output))
    n_patterns = len(network_input)


    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    print(network_input.shape)

    # normalize input
    # network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    # Integer encode notes then put an embedding layer on outputs, do I also have to change input layers as well who the hell knows.
    # print("----------------------------------------------------------------------")
    # print(network_input)
    # print(numpy.argwhere(numpy.isnan(network_input)))
    # print("----------------------------------------------------------------------")
    # print(network_output)

    return (network_input, network_output)


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    # Model 1
    # model = Sequential()
    # model.add(Embedding(n_vocab, min(600, round(1.6 * n_vocab ** 0.56)), input_length=network_input.shape[1]))
    # model.add(GRU(
    #     512,
    #     input_shape=(network_input.shape[1], network_input.shape[2]),
    #     recurrent_dropout=0.4,
    #     return_sequences=True
    # ))
    # model.add(Bidirectional(GRU(512, return_sequences=True, recurrent_dropout=0.4,)))
    # model.add(Bidirectional(GRU(512)))
    # model.add(BatchNorm())
    # model.add(Dropout(0.4))
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(BatchNorm())
    # model.add(Dropout(0.4))
    # model.add(Dense(n_vocab))
    # model.add(Activation('softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam')

    model = Sequential()
    # Embedding size for categorical variables, method found on https://forums.fast.ai/t/size-of-embedding-for-categorical-variables/42608/4
    # min(600,round(1.6*numcategories**0.56))
    print(min(600,round(1.6*n_vocab**0.56)))
    model.add(Embedding(n_vocab, min(600, round(1.6*n_vocab**0.56)),  input_length=network_input.shape[1]))
    model.add(GRU(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(512, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(512)))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model


def train(model, network_input, network_output, epoch, name, batch_size):
    """ train the neural network """
    filepath = name+"-weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=epoch, batch_size=batch_size, callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
