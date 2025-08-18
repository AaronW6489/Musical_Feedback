import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from ui_style import setup_styles, create_title, create_section
import threading
from audio_utils import record_audio
from midi_utils import get_midi_total_beats, get_midi_notes
import librosa
import numpy as np
from analysis_utils import match_notes_to_audio, performance_summary, midi_to_freq

def midi_to_note_name(midi_num):
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_num // 12) - 1
    name = names[midi_num % 12]
    return f"{name}{octave}"    

class MusicApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Instrument Performance Analyzer")
        self.geometry("900x600")
        self.configure(bg="#ecf0f1")

        setup_styles()

        self.instrument = tk.StringVar(value="piano")
        self.sheet_path = None

        self.main_frame = ttk.Frame(self)
        self.music_frame = ttk.Frame(self)

        self.canvas = None
        self.playhead = None

        self.create_widgets()

    def create_widgets(self):
        self.main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # App title
        create_title(self.main_frame, "Instrument Performance Analyzer")

        # Instrument selection section
        inst_frame = create_section(self.main_frame, "Choose Your Instrument")
        inst_frame.pack(fill="x", pady=10)
        for inst in ["piano", "guitar", "cello"]:
            ttk.Radiobutton(inst_frame, text=inst.capitalize(),
                            variable=self.instrument, value=inst).pack(anchor='w', pady=2)

        # Sheet music section
        sheet_frame = create_section(self.main_frame, "Sheet Music")
        sheet_frame.pack(fill="x", pady=10)
        ttk.Button(sheet_frame, text="Upload Sheet Music (MIDI/Image)",
                   command=self.upload_sheet).pack(pady=5)
        self.sheet_label = ttk.Label(sheet_frame, text="No file uploaded.")
        self.sheet_label.pack()

        # Recording section
        record_frame = create_section(self.main_frame, "Recording")
        record_frame.pack(fill="x", pady=10)
        ttk.Button(record_frame, text="Start Recording",
                   command=self.start_recording).pack(pady=5)
        self.feedback_label = ttk.Label(record_frame, text="Feedback will appear here.")
        self.feedback_label.pack(pady=10)

        # Music frame for piano roll display
        self.canvas_height = 600
        self.canvas = tk.Canvas(self.music_frame, width=900, height=self.canvas_height, bg='#1e272e', highlightthickness=0)
        self.canvas.pack(pady=10)
        self.music_status = ttk.Label(self.music_frame, text="", font=("Segoe UI", 12))
        self.music_status.pack()

    def upload_sheet(self):
        filetypes = [("MIDI files", "*.mid *.midi"), ("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select Sheet Music", filetypes=filetypes)
        if path:
            self.sheet_path = path
            self.sheet_label.config(text=f"Uploaded: {path.split('/')[-1]}")
        else:
            self.sheet_label.config(text="No file uploaded.")

    def start_recording(self):
        if not self.sheet_path:
            #messagebox.showwarning("No Sheet Music", "Please upload sheet music first.")
            return
        if not self.sheet_path.lower().endswith(('.mid', '.midi')):
            messagebox.showwarning("MIDI Required", "Live feedback currently requires a MIDI file.")
            return
        bpm = simpledialog.askinteger("Tempo (BPM)", "Enter the tempo (beats per minute):", minvalue=20, maxvalue=300)
        if not bpm:
            return
        total_beats = get_midi_total_beats(self.sheet_path)
        duration = total_beats * 60 / bpm
        midi_notes = get_midi_notes(self.sheet_path)
        # Visualization parameters
        keyboard_width = 120
        min_note = min((n for n,_,_ in midi_notes), default=21)  # A0
        max_note = max((n for n,_,_ in midi_notes), default=108) # C8
        note_range = max_note - min_note + 1
        # Dynamically set note_height to fit all notes in the canvas height
        max_canvas_height = 600
        note_height = max(18, min(40, max_canvas_height // note_range))
        height = note_height * note_range
        roll_width = 760
        px_per_sec = (roll_width) / 4.0  # 4 seconds visible at a time, so bars are stretched
        # Resize canvas if needed
        self.canvas.config(height=height)
        # Switch to music frame
        self.main_frame.pack_forget()
        self.music_frame.pack(fill='both', expand=True)
        self.music_status.config(text="Get ready! Recording starts soon...")
        self.update()
        def countdown_and_record():
            import time
            import sounddevice as sd
            from scipy.io.wavfile import write
            import tempfile
            chunk_sec = 0.5
            fs = 44100
            for i in range(5, 0, -1):
                self.music_status.config(text=f"Recording starts in {i}...")
                self.update()
                time.sleep(1)
            self.music_status.config(text="Recording!")
            self.update()
            self.draw_keyboard(min_note, max_note, note_height, keyboard_width, height)
            playhead = self.canvas.create_line(keyboard_width, 0, keyboard_width, height, fill='red', width=3)
            steps = int(duration * 60)
            visible_window = 4.0
            note_rects = []
            for note, start_beat, duration_beat in midi_notes:
                start_time = start_beat * 60 / bpm
                end_time = (start_beat + duration_beat) * 60 / bpm
                y = (max_note - note) * note_height
                rect = self.canvas.create_rectangle(-10, -10, -10, -10, fill='skyblue', outline='black', tags='note')
                note_rects.append((rect, start_time, end_time, y, note))
            # Real-time audio buffer
            total_samples = int(duration * fs)
            audio_buffer = np.zeros(total_samples, dtype='int16')
            chunk_samples = int(chunk_sec * fs)
            cur_sample = 0
            # Real-time feedback trackers
            live_results = []
            correct_count = 0
            total_count = 0
            def audio_callback(indata, frames, time_info, status):
                nonlocal cur_sample, audio_buffer
                if cur_sample + frames > total_samples:
                    frames = total_samples - cur_sample
                audio_buffer[cur_sample:cur_sample+frames] = indata[:frames,0]
                cur_sample += frames
            stream = sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=audio_callback)
            stream.start()
            for t in range(steps):
                cur_time = t / 60
                # Move notes
                for rect, start_time, end_time, y, note in note_rects:
                    x1 = keyboard_width + (start_time - cur_time) * px_per_sec
                    x2 = keyboard_width + (end_time - cur_time) * px_per_sec
                    if x2 < keyboard_width or x1 > keyboard_width + roll_width:
                        self.canvas.coords(rect, -10, -10, -10, -10)
                    else:
                        self.canvas.coords(rect, x1, y+4, x2, y+note_height-4)
                # Real-time analysis every chunk_sec
                if t % int(chunk_sec * 60) == 0 and cur_sample > chunk_samples:
                    chunk = audio_buffer[cur_sample-chunk_samples:cur_sample]
                    try:
                        y_chunk = chunk.astype(float) / 32768.0
                        pitches, mags = librosa.piptrack(y=y_chunk, sr=fs, fmin=30, fmax=4200)
                        detected_pitches = []
                        for i in range(pitches.shape[1]):
                            idx = np.argmax(mags[:, i])
                            freq = pitches[idx, i]
                            if mags[idx, i] > 0.1 and 30 < freq < 4200:
                                detected_pitches.append(freq)
                        # Remove duplicates and sort
                        detected_pitches = sorted(set([round(f,1) for f in detected_pitches if f > 0]))
                    except Exception:
                        detected_pitches = []
                    # Find all expected notes at this time (chord)
                    expected_notes = []
                    for note, start_beat, duration_beat in midi_notes:
                        start_time = start_beat * 60 / bpm
                        end_time = (start_beat + duration_beat) * 60 / bpm
                        if start_time <= cur_time <= end_time:
                            expected_notes.append(note)
                    msg = ""
                    if expected_notes:
                        expected_names = "+".join(midi_to_note_name(n) for n in expected_notes)
                        # For each expected note, check if a detected pitch matches (within 1 semitone)
                        corrects = []
                        for n in expected_notes:
                            efreq = midi_to_freq(n)
                            match = any(abs(np.log2(p/efreq)) < 1/12 for p in detected_pitches)
                            corrects.append(match)
                        total_count += len(expected_notes)
                        correct_count += sum(corrects)
                        detected_str = "+".join(f"{p:.0f}" for p in detected_pitches) if detected_pitches else "-"
                        msg = f"Now: {expected_names} | Detected: {detected_str} Hz\nCorrect: {sum(corrects)}/{len(corrects)} | Accuracy: {100*correct_count/total_count:.1f}%"
                    else:
                        detected_str = "+".join(f"{p:.0f}" for p in detected_pitches) if detected_pitches else "-"
                        msg = f"No note expected | Detected: {detected_str} Hz\nAccuracy: {100*correct_count/total_count:.1f}%" if total_count else ""
                    self.music_status.config(text=msg)
                self.update()
                time.sleep(1/60)
            stream.stop()
            # Save audio
            temp_dir = tempfile.gettempdir()
            wav_path = temp_dir + "/user_recording.wav"
            write(wav_path, fs, audio_buffer)
            self.music_status.config(text=f"Recording complete!\nFinal Accuracy: {100*correct_count/max(1,total_count):.1f}%")
            self.canvas.delete('all')
            self.music_frame.pack_forget()
            self.main_frame.pack(fill='both', expand=True)
            self.feedback_label.config(text=f"Recording complete! Saved to: {wav_path}\nFinal Accuracy: {100*correct_count/max(1,total_count):.1f}%")
        threading.Thread(target=countdown_and_record, daemon=True).start()

    def draw_keyboard(self, min_note, max_note, note_height, keyboard_width, height):
        self.canvas.delete('all')
        for i, note in enumerate(range(max_note, min_note-1, -1)):
            y = i * note_height
            is_white = note % 12 in [0,2,4,5,7,9,11]
            color = 'white' if is_white else 'black'
            self.canvas.create_rectangle(0, y, keyboard_width, y+note_height, fill=color, outline='gray')
            if is_white:
                note_name = midi_to_note_name(note)
                self.canvas.create_text(keyboard_width-8, y+note_height//2, text=note_name, anchor='e', fill='black', font=('Arial', 16, 'bold'))

    def draw_piano_roll(self, midi_notes, window_start, window_end, window_width, keyboard_width, height):
        self.canvas.delete('all')
        # Map MIDI notes to y positions (piano range)
        min_note = min((n for n,_,_ in midi_notes), default=60)
        max_note = max((n for n,_,_ in midi_notes), default=72)
        note_range = max_note - min_note + 1
        note_height = height // note_range
        # Draw keyboard
        for i, note in enumerate(range(max_note, min_note-1, -1)):
            y = i * note_height
            is_white = note % 12 in [0,2,4,5,7,9,11]
            color = 'white' if is_white else 'black'
            self.canvas.create_rectangle(0, y, keyboard_width, y+note_height, fill=color, outline='gray')
            if is_white:
                self.canvas.create_text(keyboard_width-10, y+note_height//2, text=str(note), anchor='e', fill='black' if is_white else 'white', font=('Arial', 8))
        # Draw piano roll window
        for note, start_beat, duration_beat in midi_notes:
            # Only draw notes in window
            note_start = start_beat
            note_end = start_beat + duration_beat
            if note_end < window_start or note_start > window_end:
                continue
            y = (max_note - note) * note_height
            # Map beats to x
            x1 = keyboard_width + int(window_width * max(0, (note_start - window_start) / (window_end - window_start)))
            x2 = keyboard_width + int(window_width * min(1, (note_end - window_start) / (window_end - window_start)))
            self.canvas.create_rectangle(x1, y+2, x2, y+note_height-2, fill='skyblue', outline='black')

if __name__ == "__main__":
    app = MusicApp()
    app.mainloop()
