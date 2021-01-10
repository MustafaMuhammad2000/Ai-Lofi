import pretty_midi
import sys
import glob

#print(sys.argv)

folder_name = sys.argv[1]

for file in glob.glob(folder_name+"/*.mid"):
    try:
        pm = pretty_midi.PrettyMIDI(file)
    except e:
        print("this file did not work: ", file)
        continue
    #print("a", pm.instruments)
    for i, v in enumerate(pm.instruments):
        print(i, v)

