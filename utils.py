import numpy as np

def midi_to_note_name(midi_num):
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_num // 12) - 1
    name = names[midi_num % 12]
    return f"{name}{octave}"

def midi_to_freq(midi_num):
    return 440.0 * (2 ** ((midi_num - 69) / 12.0))

def freq_to_midi(freq):
    return int(round(69 + 12 * np.log2(freq / 440.0)))
