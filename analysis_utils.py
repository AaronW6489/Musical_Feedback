import numpy as np
import librosa

# Helper: MIDI note number to frequency
A440 = 440.0
def midi_to_freq(midi_note):
    return A440 * 2 ** ((midi_note - 69) / 12)

def match_notes_to_audio(y, sr, midi_notes, bpm):
    """
    For each MIDI note, check if the corresponding pitch is present in the audio at the right time.
    Returns a list of (expected_note, detected_freq, correct, time_window).
    """
    results = []
    seconds_per_beat = 60 / bpm
    for note, start_beat, duration_beat in midi_notes:
        start_time = start_beat * seconds_per_beat
        end_time = (start_beat + duration_beat) * seconds_per_beat
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        chunk = y[start_sample:end_sample]
        if len(chunk) == 0:
            results.append((note, None, False, (start_time, end_time), 0.0, 0.0))
            continue
        # pitch detection
        pitches, magnitudes = librosa.piptrack(y=chunk, sr=sr)
        pitch = np.max(pitches)
        expected_freq = midi_to_freq(note)
        # Allow a tolerance of +/- 1 semitone
        correct = pitch > 0 and abs(np.log2(pitch/expected_freq)) < 1/12
        # duration detection (energy-based onset/offset estimation)
        energy = np.abs(chunk).astype(float)
        rms = np.sqrt(np.mean((energy/32768.0)**2)) if len(energy) > 0 else 0.0
        # crude duration estimate: proportion of samples above a small threshold
        thresh = np.max(np.abs(chunk)) * 0.2 if len(chunk) > 0 else 0
        active = np.sum(np.abs(chunk) > thresh)
        dur_est = (active / sr) if sr > 0 else 0.0
        results.append((note, pitch if pitch > 0 else None, correct, (start_time, end_time), dur_est, rms))
    return results

def performance_summary(results):
    total = len(results)
    correct = sum(1 for r in results if r[2])
    return {
        'total_notes': total,
        'correct_notes': correct,
        'accuracy': 100 * correct / total if total > 0 else 0
    }
