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
            print("Element is element is note", element, "then extract pitch", element.pitch)
            print("Testing duration type: ", element.duration.type, "Testing duration dots: ",element.duration.dots, "Testing quarter length", element.duration.quarterLength)
            print("Testing straight duration: ", element.duration)
            print("Testing straight offset: ", element.offset)
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print("This is notes", notes)
