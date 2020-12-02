""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


def train_network():
    """ Train a Neural Network to generate music """
    data = get_notes()['data']
    # offset = get_notes()['offset']
    # duration = get_notes()['duration']

    # get amount of pitch names
    data_vocab = len(set(data))
    print(data_vocab)
    # n_vocab = len(set(notes))
    # o_vocab = len(set(offset))
    # d_vocab = len(set(duration))

    network_input, network_output = prepare_sequences(data, data_vocab)
    # network_input_notes, network_output_notes = prepare_sequences(notes, n_vocab)
    # network_input_offset, network_output_offset = prepare_sequences(offset, o_vocab)
    # network_input_duration, network_output_duration = prepare_sequences(duration, d_vocab)

    model = create_network(network_input, data_vocab)
    # n_model = create_network(network_input_notes, n_vocab)
    # o_model = create_network(network_input_offset, o_vocab)
    # d_model = create_network(network_input_duration, d_vocab)

    train(model, network_input, network_output, 2000, "Data")
    # train(n_model, network_input_notes, network_output_notes, 100, "Notes")
    # train(o_model, network_input_offset, network_output_offset, 200, "Offset")
    # train(d_model, network_input_duration, network_output_duration, 100, "Duration")


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    data = []
    # notes = []
    # offset = []
    # duration = []

    for file in glob.glob("Animelofi_midi/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                data.append((str(element.pitch), str(element.offset), str(element.duration.quarterLength)))
                # notes.append(str(element.pitch))
                # offset.append(str(element.offset))
                # duration.append(str(element.duration.quarterLength))
            elif isinstance(element, chord.Chord):
                data.append(('.'.join(str(n) for n in element.normalOrder), str(element.offset), str(element.duration.quarterLength)))
                # notes.append('.'.join(str(n) for n in element.normalOrder))
                # offset.append(str(element.offset))
                # duration.append(str(element.duration.quarterLength))

    # with open('data/notes', 'wb') as filepath:
    #     pickle.dump(notes, filepath)
    #
    # with open('data/offset', 'wb') as filepath:
    #     pickle.dump(offset, filepath)
    #
    # with open('data/duration', 'wb') as filepath:
    #     pickle.dump(duration, filepath)

    with open('data/data', 'wb') as filepath:
        pickle.dump(data, filepath)

    # return {'notes':notes, 'offset':offset, 'duration':duration}
    return {'data': data}
    # return data


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    # Model 1
    # model = Sequential()
    # model.add(GRU(
    #     512,
    #     input_shape=(network_input.shape[1], network_input.shape[2]),
    #     recurrent_dropout=0.3,
    #     return_sequences=True
    # ))
    # model.add(GRU(512, return_sequences=True, recurrent_dropout=0.3,))
    # model.add(GRU(512))
    # model.add(BatchNorm())
    # model.add(Dropout(0.3))
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(BatchNorm())
    # model.add(Dropout(0.3))
    # model.add(Dense(n_vocab))
    # model.add(Activation('softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model = Sequential()
    model.add(GRU(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(512, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(512)))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def train(model, network_input, network_output, epoch, name):
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

    model.fit(network_input, network_output, epochs=epoch, batch_size=512, callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
