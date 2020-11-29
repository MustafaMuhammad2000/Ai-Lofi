import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord

notes = []
for file in glob.glob("lofi_midi/*.mid"):
    midi = converter.parse(file)

    notes_to_parse = None

    try:  # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            print("Testing quarter length", element.duration.quarterLength, type(element.duration.quarterLength))
            print(element.duration)
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print("This is notes", notes)
