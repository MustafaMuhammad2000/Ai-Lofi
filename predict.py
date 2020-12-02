""" This module generates notes for a midi file using the
    trained neural network """
import pickle
from fractions import Fraction
import numpy
import os
from music21 import instrument, note, stream, chord
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import GRU 
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('data/data', 'rb') as filepath:
        data = pickle.load(filepath)
    # with open('data/notes', 'rb') as filepath:
    #     notes = pickle.load(filepath)
    # with open('data/offset', 'rb') as filepath:
    #     offset = pickle.load(filepath)
    # with open('data/duration', 'rb') as filepath:
    #     duration = pickle.load(filepath)


    # Get all pitch names
    datanames = sorted(set(item for item in data))

    # pitchnames = sorted(set(item for item in notes))
    # offsettimes = sorted(set(item for item in offset))
    # durationtimes = sorted(set(item for item in duration))

    print("num data", len(datanames))

    # print("num pitch", len(pitchnames))
    # # print("num offset", len(offsettimes))
    # print("num durration", len(durationtimes))

    # Get all pitch names
    vocab = len(set(data))

    # n_vocab = len(set(notes))
    # # o_vocab = len(set(offset))
    # d_vocab = len(set(duration))

    network_input, normalized_input = prepare_sequences(data,datanames,vocab)

    # network_input_notes, normalized_notes = prepare_sequences(notes, pitchnames, n_vocab)
    # # network_input_offset, normalized_offset = prepare_sequences(offset, offsettimes, o_vocab)
    # network_input_duration, normalized_duration = prepare_sequences(duration, durationtimes, d_vocab)
    #print(network_input_notes)


    model = create_network(normalized_input, vocab, 'Data-weights-improvement-288-0.1145-bigger.hdf5')

    # n_model = create_network( normalized_notes, n_vocab,'Notes-weights-improvement-89-0.0489-bigger.hdf5')
    # # o_model = create_network( normalized_offset, o_vocab,'Offset-weights-improvement-140-0.2950-bigger.hdf5')
    # d_model = create_network( normalized_duration, d_vocab,'Duration-weights-improvement-94-0.0243-bigger.hdf5')

    start = numpy.random.randint(0, len(network_input)-1)

    print("This is start: ",start)
    prediction_data = generate_notes(model, network_input, datanames, vocab, start)

    # print("==================== notes =========================")
    # prediction_notes = generate_notes(n_model, network_input_notes, pitchnames, n_vocab, start)
    # print("==================== offset =========================")
    # # prediction_offsets = generate_notes(o_model, network_input_offset, offsettimes, o_vocab, start)
    # print("==================== duration =========================")
    # prediction_durations = generate_notes(d_model,network_input_duration, durationtimes, d_vocab, start)

    create_midi(prediction_data, None, None)


def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)
    print("Length of network input",len(network_input))
    print("Length of one network input", len(network_input[0]))
    return (network_input, normalized_input)


def create_network(network_input, n_vocab, weights):
    """ create the structure of the neural network """
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

    # Load the weights to each node
    model.load_weights(weights)

    return model


def generate_notes(model, network_input, pitchnames, n_vocab, start):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    # start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    # print(int_to_note)

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        # print("One prediction input",prediction_input)
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        # print(prediction)
        # print("index", index)
        result = int_to_note[index]
        # print("result",result)
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print("This is prediction output",prediction_output)
    return prediction_output


def create_midi(prediction_notes, prediction_offset, prediction_duration):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []
    
    # if(len(prediction_notes)!=len(prediction_duration)):
    #     print("create midi input args not same length")
    #     os.exit(-1)

    # create note and chord objects based on the values generated by the model
    for i, pattern in enumerate(prediction_notes):
        print(pattern)
        # pattern is a chord

        if ('.' in pattern[0]) or pattern[0].isdigit():
            notes_in_chord = pattern[0].split('.')
            notes = []
            for current_note in notes_in_chord:
                # dur = float(Fraction(prediction_duration[i]))
                #if(isinstance(dur, fractions.Fraction):

                #if(type(dur)==fractions.Fraction):
                # if(dur > 1):
                #     dur = 1
                new_note = note.Note(int(current_note), quarterLength=float(Fraction(pattern[2])))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = float(Fraction(pattern[1]))
            # new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            # dur = float(Fraction(prediction_duration[i]))
            # if (dur > 1):
            #     dur = 1
            new_note = note.Note(pattern[0], quarterLength=float(Fraction(pattern[2])))
            # new_note.offset = offset
            new_note.offset = float(Fraction(pattern[1]))
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        # offset += dur
    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')


if __name__ == '__main__':
    generate()
