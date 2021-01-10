""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
import pretty_midi
from fractions import Fraction
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.layers import Activation
from keras.layers import Embedding
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

def train_network():
    """ Train a Neural Network to generate music """
    pitch, velocity, offset, duration, songlengths = get_notes_multiple()

    # get amount of pitch names
    print("+++++++++++++++++++")
    p_vocab = len(set(pitch))
    print(p_vocab)
    v_vocab = len(set(velocity))
    print(v_vocab)
    o_vocab = len(set(offset))
    print(o_vocab)
    d_vocab = len(set(duration))
    print(d_vocab)


    # network_input_pitch, network_output_pitch = prepare_sequences(pitch, p_vocab, songlengths)
    # p_model = create_network(network_input_pitch, p_vocab)
    # train(p_model, network_input_pitch, network_output_pitch, 100, "Pitches", 256)
    # #
    #
    # network_input_velocity, network_output_velocity = prepare_sequences(velocity, v_vocab, songlengths)
    # v_model = create_network(network_input_velocity, v_vocab)
    # train(v_model, network_input_velocity, network_output_velocity, 100, "Velocities", 256)
    # #
    #
    # network_input_offset, network_output_offset = prepare_sequences(offset, o_vocab, songlengths)
    # o_model = create_network(network_input_offset, o_vocab)
    # train(o_model, network_input_offset, network_output_offset, 150, "Offset", 256)
    #

    network_input_duration, network_output_duration = prepare_sequences(duration, d_vocab, songlengths)
    d_model = create_network(network_input_duration, d_vocab)
    train(d_model, network_input_duration, network_output_duration, 150, "Duration", 256)
    #

    # model = create_network(network_input, data_vocab)
    # n_model = create_network(network_input_notes, n_vocab)
    # o_model = create_network(network_input_offset, o_vocab)
    # d_model = create_network(network_input_duration, d_vocab)
    # v_model = create_network(network_input_velocity, v_vocab)
    #

    # train(model, network_input, network_output, 5000, "Data")
    # train(n_model, network_input_notes, network_output_notes, 150, "Notes")
    # train(o_model, network_input_offset, network_output_offset, 100, "Offset")
    # train(d_model, network_input_duration, network_output_duration, 125, "Duration")


def get_notes_multiple():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    velocity = []
    pitch = []
    offset = []
    duration = []
    songlengths = []
    BadFiles = 0
    GoodFiles = 0
    for file in glob.glob("Yuge_Data_Anime2/*.mid"):
        print("--------------------------------------")
        print("Parsing %s" % file)
        print("--------------------------------------")
        notes = []
        notesVar = []

        try:
            pm = pretty_midi.PrettyMIDI(file)
            GoodFiles += 1
        except:
            print("This file did not work: ", file)
            BadFiles += 1
            continue

        for i in range(0, len(pm.instruments)):
            pm.instruments[i].remove_invalid_notes()
            if pm.instruments[i].program != 0: print(pretty_midi.program_to_instrument_name(pm.instruments[i].program))
            print(len(pm.instruments[i].notes))
            notes.append(pm.instruments[i].notes)
            if len(pm.instruments[i].pitch_bends) > 0: print("Pitch Bends", len(pm.instruments[i].pitch_bends))
            if len(pm.instruments[i].control_changes) > 0: print("Control changes", len(pm.instruments[i].control_changes))

        for list in notes:
            length = 0
            for note in list:
                notesVar.append((note.start, note.get_duration(), note.pitch, note.velocity))
                length += 1
            notesVar = sorted(notesVar)
            prevOffset = 0
            songlengths.append(length)
            for note in notesVar:
                offset.append(round(note[0] - prevOffset, 1))
                duration.append(round(note[1], 1))
                pitch.append(note[2])
                velocity.append(note[3])
                prevOffset = note[0]
            notesVar = []

    # print("Offset")
    # print(len(offset))
    # print("Unique Offset", len(set(offset)))
    # print(sorted(set(offset)))
    # print("Duration")
    # print(len(duration))
    # print("Unique Duration", len(set(duration)))
    # print(sorted(set(duration)))
    # print("Notes")
    # print(len(pitch))
    # print("Unique Pitches", len(set(pitch)))
    # print("Notes")
    # print(len(velocity))
    # print("Unique Velocity", len(set(velocity)))
    # print("This many files did work", GoodFiles)
    # print("This many files did not work", BadFiles)
    # print(len(songlengths))
    # print(songlengths)

    with open('data/pitch', 'wb') as filepath:
        pickle.dump(pitch, filepath)
    with open('data/offset', 'wb') as filepath:
        pickle.dump(offset, filepath)
    with open('data/duration', 'wb') as filepath:
        pickle.dump(duration, filepath)
    with open('data/velocity', 'wb') as filepath:
        pickle.dump(velocity, filepath)
    with open('data/songlengths', 'wb') as filepath:
        pickle.dump(songlengths, filepath)

    return pitch, velocity, offset, duration, songlengths


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
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    # j = 0
    # i = 0
    # total = songlengths[j]
    # while i < len(notes)-sequence_length:
    #     if(sequence_length+i <= total):
    #         sequence_in = notes[i:i + sequence_length]
    #         sequence_out = notes[i + sequence_length]
    #         # print([note_to_int[i] for i in sequence_in])
    #         network_input.append([note_to_int[char] for char in sequence_in])
    #         network_output.append(note_to_int[sequence_out])
    #         i = i+1
    #     else:
    #         i = total
    #         j = j+1
    #         total = total+songlengths[j]
    # print(j)


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
    model = Sequential()
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
