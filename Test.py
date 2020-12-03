import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord

notes = []
for file in glob.glob("Animelofi_midi/*.mid"):
    midi = converter.parse(file)

    notes_to_parse = None

    try:  # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            # notes.append([str(element.pitch),str(element.offset), str(element.duration.quarterLength)])
            notes.append(str(element.pitch))
            notes.append(str(element.offset))
            notes.append(str(element.duration.quarterLength))
        elif isinstance(element, chord.Chord):
            # notes.append(['.'.join(str(n) for n in element.normalOrder), str(element.offset), str(element.duration.quarterLength)])
            notes.append('.'.join(str(n) for n in element.normalOrder))
            notes.append(str(element.offset))
            notes.append(str(element.duration.quarterLength))

print("This is notes", notes)
print(len(notes))