""" This module generates notes for a midi file using the
    trained neural network """
import pickle
from fractions import Fraction
import numpy
import os
import glob
import pretty_midi
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import Embedding
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
    # load the notes used to train the model
    # with open('data/data', 'rb') as filepath:
    #     data = pickle.load(filepath)
    with open('data/songlengths', 'rb') as filepath:
        songlengths = pickle.load(filepath)
    with open('data/pitch', 'rb') as filepath:
        pitch = pickle.load(filepath)
    with open('data/velocity', 'rb' )as filepath:
        velocity = pickle.load(filepath)
    with open('data/offset', 'rb') as filepath:
        offset = pickle.load(filepath)
    with open('data/duration', 'rb') as filepath:
        duration = pickle.load(filepath)

    # Get all pitch names
    # datanames = sorted(set(item for item in data))
    pitchnames = sorted(set(item for item in pitch))
    velocitynames = sorted(set(item for item in velocity))
    offsettimes = sorted(set(item for item in offset))
    durationtimes = sorted(set(item for item in duration))

    # print("num data", len(datanames))

    print("num pitch", len(pitchnames))
    print("num velocity", len(velocitynames))
    print("num offset", len(offsettimes))
    print("num duration", len(durationtimes))

    # Get all pitch names
    # data_vocab = len(set(data))
    # print("num data2", data_vocab)

    p_vocab = len(set(pitch))
    v_vocab = len(set(velocity))
    o_vocab = len(set(offset))
    d_vocab = len(set(duration))

    pred_pitch, pred_velocity, pred_offset, pred_duration, pred_songlengths = load_song("Bleach - Will of the Heart.mid")
    print("part length = ", pred_songlengths)

    pred_network_input_pitch, pred_normalized_pitch, pitchPieceLength = prepare_sequences(pred_pitch, pitchnames, p_vocab, pred_songlengths)
    print(pitchPieceLength)
    pred_network_input_velocity, pred_normalized_velocity, Ff = prepare_sequences(pred_velocity, velocitynames, v_vocab, pred_songlengths)
    pred_network_input_offset, pred_normalized_offset, FFf = prepare_sequences(pred_offset, offsettimes, o_vocab, pred_songlengths)
    pred_network_input_duration, pred_normalized_duration, FFFf = prepare_sequences(pred_duration, durationtimes, d_vocab, pred_songlengths)

    p_model = create_network(pred_normalized_pitch, p_vocab, 'Pitches-weights-improvement-56-0.7409-bigger.hdf5')
    v_model = create_network(pred_normalized_velocity, v_vocab, 'Velocities-weights-improvement-99-0.2128-bigger.hdf5')
    o_model = create_network(pred_normalized_offset, o_vocab, 'Offset-weights-improvement-78-0.2184-bigger.hdf5')
    d_model = create_network(pred_normalized_duration, d_vocab, 'Duration-weights-improvement-42-0.4166-bigger.hdf5')

    # TRY FORCING THE SONG INSTEAD OF USING BLACKBOX MUSIC YA KNOW MA MANS HOWS IT GOING!!
    # start2 = numpy.random.randint(0, len(pred_notes)-1)
    # start2 = 0
    # print("==================== pitch =========================")
    # prediction_pitches = generate_notes(p_model, pred_network_input_pitch, pitchnames, start)
    # print("==================== velocity =========================")
    # prediction_velocites = generate_notes(v_model, pred_network_input_velocity, pitchnames, start)
    # print("==================== offset =========================")
    # prediction_offsets = generate_notes(o_model, pred_network_input_offset, offsettimes, start)
    # print("==================== duration =========================")
    # prediction_durations = generate_notes(d_model, pred_network_input_duration, durationtimes, start)

    piecesPitches = []
    piecesVelocties = []
    piecesOffsets = []
    piecesDuration = []

    start = 0
    for i in range(0, len(pitchPieceLength)):
        piecesPitches.append(pred_network_input_pitch[start:start+pitchPieceLength[i]])
        piecesVelocties.append(pred_network_input_velocity[start:start+pitchPieceLength[i]])
        piecesOffsets.append(pred_network_input_offset[start:start+pitchPieceLength[i]])
        piecesDuration.append(pred_network_input_duration[start:start+pitchPieceLength[i]])
        start += pitchPieceLength[i]
        # print(start)

    print(len(piecesPitches))
    print(len(piecesPitches[0]))


    prediction_pitches = []
    prediction_velocites = []
    prediction_offsets = []
    prediction_durations = []
    for i in range(0, len(piecesPitches)):
        print("==================== pitch =========================")
        prediction_pitches.append(generate_notes(p_model, piecesPitches[i], pitchnames, 0))
        print("==================== velocity =========================")
        prediction_velocites.append(generate_notes(v_model, piecesVelocties[i], velocitynames, 0))
        print("==================== offset =========================")
        prediction_offsets.append(generate_notes(o_model, piecesOffsets[i], offsettimes, 0))
        print("==================== duration =========================")
        prediction_durations.append(generate_notes(d_model, piecesDuration[i], durationtimes, 0))

    print(prediction_pitches)
    print(type(prediction_pitches))

    print(len(prediction_offsets))
    create_midi(prediction_pitches, prediction_velocites, prediction_offsets, prediction_durations)


def prepare_sequences(notes, pitchnames, n_vocab, songlengths):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    sequence_length = 32

    pieceLength = []
    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    output = []
    # for i in range(0, len(notes) - sequence_length, 1):
    #     sequence_in = notes[i:i + sequence_length]
    #     sequence_out = notes[i + sequence_length]
    #     network_input.append([note_to_int[char] for char in sequence_in])
    #     output.append(note_to_int[sequence_out])

    j = 0
    i = 0
    total = songlengths[j]
    length = 0
    while i < len(notes) - sequence_length:
        if (sequence_length + i < total):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            output.append(note_to_int[sequence_out])
            i = i + 1
            length += 1
        else:
            i = total
            j = j + 1
            total = total + songlengths[j]
            pieceLength.append(length)
            length = 0
    pieceLength.append(length)

    print(j)
    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))

    # normalize input
    # normalized_input = normalized_input / float(n_vocab)

    print("Length of network input", len(network_input))
    # print("Length of one network input", len(network_input[0]))

    return network_input, normalized_input, pieceLength


def load_song(filename):
    velocity = []
    pitch = []
    offset = []
    duration = []
    songlengths = []
    notes = []
    notesVar = []

    pm = pretty_midi.PrettyMIDI(filename)

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

    return pitch, velocity, offset, duration, songlengths

def create_network(network_input, n_vocab, weights):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(Embedding(n_vocab, min(600, round(1.6 * n_vocab ** 0.56)), input_length=network_input.shape[1]))
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
    # Load the weights to each node
    model.load_weights(weights)

    return model


def generate_notes(model, network_input, pitchnames, start):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    # start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    # print(int_to_note)

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(330):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))

        # print("One prediction input",prediction_input)
        # prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        # print(prediction)

        ind = numpy.argpartition(prediction[0], -3)[-3:]
        # print(ind)

        index = numpy.argmax(prediction)
        print(prediction[0][index])
        # print("Ind 0: ",prediction[0][ind[0]])
        # print("Ind 1: ", prediction[0][ind[1]])
        # print("Ind 2: ", prediction[0][ind[2]])
        # print(index)
        # print(prediction)
        # print("index", index)
        result = int_to_note[index]
        # print("result",result)
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    # print("This is prediction output",prediction_output)
    return prediction_output


def generate_notes2(model, network_input, pitchnames):
    """ Generate notes from the neural network based on a sequence of notes """

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    # pattern = network_input[start]

    prediction_output = []

    # generate 500 notes
    i = 0
    while i < len(network_input):
        prediction_input = numpy.reshape(network_input[i], (1, len(network_input[i]), 1))
        prediction = model.predict(prediction_input, verbose=0)
        index = numpy.argmax(prediction)
        print(prediction[0][index])
        result = int_to_note[index]
        prediction_output.append(result)
        i += 1
    print(i)
    # for note_index in range(500):
    #     prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
    #
    #     # print("One prediction input",prediction_input)
    #     prediction = model.predict(prediction_input, verbose=0)
    #     index = numpy.argmax(prediction)
    #     print(prediction[0][index])
    #     result = int_to_note[index]
    #     prediction_output.append(result)
    #     pattern.append(index)
    #     pattern = pattern[1:len(pattern)]

    # print("This is prediction output",prediction_output)
    return prediction_output


def create_midi(prediction_pitches, prediction_velocity, prediction_offset, prediction_duration):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    sample_Output = pretty_midi.PrettyMIDI()
    AcousticPiano = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    for i in range(0, len(prediction_pitches)):
        piano = pretty_midi.Instrument(program=AcousticPiano)
        time = 0
        for j in range(0, len(prediction_pitches[i])):
            time += prediction_offset[i][j]
            note = pretty_midi.Note(pitch=prediction_pitches[i][j], velocity=prediction_velocity[i][j], start=time,end=time+prediction_duration[i][j])
            piano.notes.append(note)
            print(note)
        sample_Output.instruments.append(piano)

    sample_Output.write("test_output_pretty_midi.mid")


if __name__ == '__main__':
    generate()
