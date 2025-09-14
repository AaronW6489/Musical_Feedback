from mido import MidiFile

def get_midi_total_beats(midi_path):
    """
    Returns the total number of beats in the MIDI file.
    """
    mid = MidiFile(midi_path)
    total_ticks = 0
    ticks_per_beat = mid.ticks_per_beat
    for track in mid.tracks:
        for msg in track:
            if msg.type in ('note_on', 'note_off', 'set_tempo'):
                total_ticks += msg.time
    total_beats = total_ticks / ticks_per_beat
    return total_beats

def get_midi_tempo(midi_path):
    """
    Extract the first tempo (microseconds per beat) from the MIDI file and return BPM.
    If no tempo message exists, return a sensible default (120 BPM).
    """
    mid = MidiFile(midi_path)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                # mido gives tempo in microseconds per beat
                mpb = msg.tempo
                bpm = 60000000 / mpb if mpb > 0 else 120
                return int(round(bpm))
    # fallback default
    return 120

# Optionally, you can extract the note sequence for further analysis

def get_midi_notes(midi_path):
    """
    Returns a list of (note, start_beat, duration_beat) tuples from the MIDI file.
    """
    mid = MidiFile(midi_path)
    notes = []
    ticks_per_beat = mid.ticks_per_beat
    for track in mid.tracks:
        abs_time = 0
        note_on_times = {}
        for msg in track:
            abs_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_on_times[msg.note] = abs_time
            elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and msg.note in note_on_times:
                start = note_on_times.pop(msg.note)
                duration = abs_time - start
                notes.append((msg.note, start / ticks_per_beat, duration / ticks_per_beat))
    return notes
