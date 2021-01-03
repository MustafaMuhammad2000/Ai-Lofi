import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord, stream
import pretty_midi
from fractions import Fraction

# Single Music 21 File
def music21Single(filename):
    notes = []
    duration = []
    offset = []
    midi = converter.parse(filename)
    print("Parsing %s" % filename)

    notes_to_parse = None

    try:  # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        choice = 0
        print("Music 21 File", s2.parts[choice], len(s2.parts[choice]), len(s2.parts))
        if (len(s2.parts) > 1):
            for i in range(1, len(s2.parts)):
                print("Music 21 File", len(s2.parts[i]))
                if (len(s2.parts[choice]) < len(s2.parts[i])):
                    choice = i
        # print("Music 21 File", s2.parts[choice], len(s2.parts), len(s2.parts[choice]))
        notes_to_parse = s2.parts[choice].recurse()
    except:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    prevOffset = 0
    for element in notes_to_parse:
        currOffset = float(Fraction(element.offset - prevOffset))
        currOffset = round(currOffset, 5)
        if (currOffset > 5):
            currOffset = 5

        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
            offset.append(str(currOffset))
            duration.append(str(element.duration.quarterLength))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
            offset.append(str(currOffset))
            duration.append(str(element.duration.quarterLength))
        prevOffset = element.offset
    print(len(notes))

    output_notes = []
    numoffset = 0
    for i, pattern in enumerate(notes):
        # print(pattern)
        # pattern is a chord

        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note), quarterLength=float(Fraction(duration[i])))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = numoffset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern, quarterLength=float(Fraction(duration[i])))
            new_note.offset = numoffset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        numoffset += float(Fraction(offset[i]))
    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output_music21.mid')
    print("--------------------------------------------------")


# notes = []
# pitches = []
# offset = []
# duration = []
# songlength = []
# zeros = 0
# totalsongs = 0
# for file in glob.glob("Yuger_Data/*.mid"):
#     midi = converter.parse(file)
#
#     notes_to_parse = None
#     totalsongs = totalsongs+1
#     # print("Parsing %s" % file)
#
#     try:  # file has instrument part
#         s2 = instrument.partitionByInstrument(midi)
#         choice = 0
#         # print("length of S2 parts", len(s2.parts))
#         # print("What instrument s2 has", s2.parts[len(s2.parts)-1])
#         if(len(s2.parts) > 1):
#             # print(file, len(s2.parts))
#             for i in range(0, len(s2.parts)):
#                 # if(len(s2.parts[i]) > 1):
#                 print(s2.parts[i], len(s2.parts[i]))
#                 if(len(s2.parts[choice]) < len(s2.parts[i])):
#                     choice = i
#             # print("This is decision",len(s2.parts[choice]))
#         # print(s2.parts[choice], len(s2.parts), len(s2.parts[choice]))
#         notes_to_parse = s2.parts[choice].recurse()
#
#     except:  # file has notes in a flat structure
#         # print("Hello")
#         notes_to_parse = midi.flat.notes
#
#     prevOffset = 0
#     length = 0
#     fileNotes = []
#     fileOffset = []
#     fileDuration = []
#     for element in notes_to_parse:
#         # print(element.offset, prevOffset)
#         currOffset = float(Fraction(element.offset-prevOffset))
#         currOffset = round(currOffset, 3)
#         # float(Fraction(prediction_offset[i]))
#         if(currOffset > 5):
#             currOffset = 5
#         # print(currOffset)
#         if isinstance(element, note.Note):
#             length = length+1
#             fileNotes.append(str(element.pitch))
#             fileOffset.append(str(element.offset))
#             fileDuration.append(str(element.duration.quarterLength))
#             # notes.append([str(element.pitch),str(currOffset), str(element.duration.quarterLength)])
#             # notes.append(str(element.pitch))
#             # notes.append(str(element.offset))
#             # notes.append(str(element.duration.quarterLength))
#         elif isinstance(element, chord.Chord):
#             length = length+1
#             fileNotes.append('.'.join(str(n) for n in element.normalOrder))
#             fileOffset.append(str(element.offset))
#             fileDuration.append(str(element.duration.quarterLength))
#             # notes.append(['.'.join(str(n) for n in element.normalOrder), str(currOffset), str(element.duration.quarterLength)])
#             # notes.append('.'.join(str(n) for n in element.normalOrder))
#             # notes.append(str(element.offset))
#             # notes.append(str(element.duration.quarterLength))
#         prevOffset = element.offset
#     if length == 0:
#         zeros = zeros+1
#         print(file)
#
#     songlength.append(length)
#     pitches.append(fileNotes)
#     offset.append(fileOffset)
#     duration.append(fileDuration)
#
#
#
#
# print("This is how many files suck ass", zeros)
# print(len(pitches))
# print(len(offset))
# print(len(duration))
#
# for i in range(0, len(pitches)):
#     # print("Index: ",i, " Length of notes: ",len(pitches[i]))
#     # print("Index: ",i, " Length of offset: ", len(offset[i]))
#     # print("Index: ",i, " Length of duration: ", len(duration[i]))
#     if(len(pitches[i]) != len(duration[i]) or len(pitches[i]) != len(offset[i])):
#         print('---------------HELLOOOOOOOO---------------')
#
# print('Whats popping')
#
# # # print("This is notes", notes)
# # print("This is len Notes", len(notes))
# # print("This is length of number of songs", len(songlength))
# # print("This is the number of files",totalsongs)
# # print("This is length of each song", songlength)
# # print("This is average length of all songs", sum(songlength)/len(songlength))
# #
# lessthan32 = sum(i < 32 for i in songlength)
# print("Songs less than 100 notes long", lessthan100)


# Single Pretty_Midi FILE
def prettyMidiSingle(filename):
    pm = pretty_midi.PrettyMIDI(filename)
    notes = []
    for i in range(0, len(pm.instruments)):
        pm.instruments[i].remove_invalid_notes()
        print(pretty_midi.program_to_instrument_name(pm.instruments[i].program))
        print(len(pm.instruments[i].notes))
        print(pm.instruments[i].notes)
        notes.append(pm.instruments[i].notes)
        print("Pitch Bends", len(pm.instruments[0].pitch_bends))
        print("Control changes", len(pm.instruments[0].control_changes))
    print(len(notes))

    sample_Output = pretty_midi.PrettyMIDI()
    AcousticPiano = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano1 = pretty_midi.Instrument(program=AcousticPiano)
    piano2 = pretty_midi.Instrument(program=AcousticPiano)
    for note in notes[0]:
        piano1.notes.append(note)
    for note in notes[1]:
        piano2.notes.append(note)

    sample_Output.instruments.append(piano1)
    sample_Output.instruments.append(piano2)
    sample_Output.write("test_output_pretty_midi.mid")
    print("--------------------------------------------------")

# Multiple Pretty_Midi File
def prettyMidiMultiple():
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
            GoodFiles +=1
        except:
            print("This file did not work: ", file)
            BadFiles +=1
            continue

        for i in range(0, len(pm.instruments)):
            pm.instruments[i].remove_invalid_notes()
            if pm.instruments[i].program != 0: print(pretty_midi.program_to_instrument_name(pm.instruments[i].program))
            print(len(pm.instruments[i].notes))
            notes.append(pm.instruments[i].notes)
            # if len(pm.instruments[i].pitch_bends) > 0: print("Pitch Bends", len(pm.instruments[i].pitch_bends))
            # if len(pm.instruments[i].control_changes) > 0: print("Control changes", len(pm.instruments[i].control_changes))

        for list in notes:
            length = 0
            for note in list:
                notesVar.append((note.start, note.get_duration(), note.pitch, note.velocity))
                length +=1
            notesVar = sorted(notesVar)
            prevOffset = 0
            songlengths.append(length)
            for note in notesVar:
                offset.append(round(note[0]-prevOffset, 1))
                duration.append(round(note[1], 1))
                pitch.append(note[2])
                velocity.append(note[3])
                prevOffset = note[0]
            notesVar = []
    print("Offset")
    print(len(offset))
    print("Unique Offset", len(set(offset)))
    print(sorted(set(offset)))
    print("Duration")
    print(len(duration))
    print("Unique Duration", len(set(duration)))
    print(sorted(set(duration)))
    print("Notes")
    print(len(pitch))
    print("Unique Pitches", len(set(pitch)))
    print("Notes")
    print(len(velocity))
    print("Unique Velocity", len(set(velocity)))
    print("This many files did work", GoodFiles)
    print("This many files did not work", BadFiles)
    print(len(songlengths))
    print(songlengths)
    # velocity = []
    # pitch = []
    # offset = []
    # duration = []
    # notes = []
    # notesVar = []
    #
    # pm = pretty_midi.PrettyMIDI("Joe Hisaishi - One Summer's Day (Spirited Away).mid")
    #
    # for i in range(0, len(pm.instruments)):
    #     pm.instruments[i].remove_invalid_notes()
    #     print(pretty_midi.program_to_instrument_name(pm.instruments[i].program))
    #     print(len(pm.instruments[i].notes))
    #     print(pm.instruments[i].notes)
    #     notes.append(pm.instruments[i].notes)
    #     print("Pitch Bends", len(pm.instruments[0].pitch_bends))
    #     print("Control changes", len(pm.instruments[0].control_changes))
    #
    # for list in notes:
    #     for note in list:
    #         notesVar.append((note.start, note.get_duration(), note.pitch, note.velocity))
    #     notesVar = sorted(notesVar)
    #     print(len(notesVar), notesVar)
    #     for note in notesVar:
    #         offset.append(note[0])
    #         duration.append(note[1])
    #         pitch.append(note[2])
    #         velocity.append(note[3])
    #     notesVar = []
    #
    # print(offset)
    # print(len(offset))
    # print(velocity)
    # print(len(velocity))


if __name__ == '__main__':
    # prettyMidiSingle("Joe Hisaishi - One Summer's Day (Spirited Away).mid")
    # music21Single("Joe Hisaishi - One Summer's Day (Spirited Away).mid")
    prettyMidiMultiple()
