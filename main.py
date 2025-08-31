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

def freq_to_note_name(freq):
    """Convert frequency (Hz) to nearest MIDI note name."""
    try:
        midi = int(round(69 + 12 * np.log2(freq / 440.0)))
        return midi_to_note_name(midi)
    except Exception:
        return None

class MusicApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Instrument Performance Analyzer")
        self.geometry("900x600")
        self.configure(bg="#ecf0f1")

        setup_styles()

        # warm up pitch detection in background to avoid UI freeze later
        threading.Thread(target=self._warmup_pitch_detection, daemon=True).start()

        self.instrument = tk.StringVar(value="piano")
        self.sheet_path = None

        self.main_frame = ttk.Frame(self)
        self.music_frame = ttk.Frame(self)

        self.canvas = None
        self.playhead = None

        self.create_widgets()

    def _warmup_pitch_detection(self):
        """Call a lightweight piptrack operation to import heavy deps and warm up numba/librosa."""
        try:
            small = np.zeros(1024, dtype=float)
            # small warmup; ignore results
            _p, _m = librosa.piptrack(y=small, sr=22050)
        except Exception:
            pass

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
            from scipy.io.wavfile import write, read
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
            evaluated = set()
            evaluation_results = {}  # (note, start_time) -> bool
            for note, start_beat, duration_beat in midi_notes:
                start_time = start_beat * 60 / bpm
                end_time = (start_beat + duration_beat) * 60 / bpm
                y = (max_note - note) * note_height
                rect = self.canvas.create_rectangle(-10, -10, -10, -10, fill='deepskyblue', outline='black', tags='note')
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
                            if mags[idx, i] > 0.05 and 30 < freq < 4200:
                                detected_pitches.append(freq)
                        # map to nearest midi note names and deduplicate
                        detected_pitches = [round(f,1) for f in detected_pitches if f > 0]
                        detected_names = []
                        seen = set()
                        for fp in detected_pitches:
                            nm = freq_to_note_name(fp)
                            if nm and nm not in seen:
                                seen.add(nm)
                                detected_names.append(nm)
                    except Exception:
                        detected_pitches = []
                        detected_names = []
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
                        # Update colors for evaluated notes when their start time has passed
                        for rect, st, et, y, note in note_rects:
                            if note in expected_notes and (note, st) not in evaluated and cur_time >= st:
                                # find match for this note
                                efreq = midi_to_freq(note)
                                is_match = any(abs(np.log2(p/efreq)) < 1/12 for p in detected_pitches)
                                color = 'green' if is_match else 'red'
                                try:
                                    self.canvas.itemconfig(rect, fill=color)
                                except Exception:
                                    pass
                                evaluated.add((note, st))
                                evaluation_results[(note, st)] = is_match
                        total_count += len(expected_notes)
                        correct_count += sum(corrects)
                        detected_str = "+".join(detected_names) if detected_names else "-"
                        msg = f"Now: {expected_names} | Detected: {detected_str}\nCorrect: {sum(corrects)}/{len(corrects)} | Accuracy: {100*correct_count/total_count:.1f}%"
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
            # After recording, run playback review: replay the recorded audio and the MIDI with colored feedback
            try:
                self.playback_and_review(wav_path, midi_notes, bpm, evaluation_results, note_rects, keyboard_width, note_height, roll_width, duration)
            except Exception:
                pass
        threading.Thread(target=countdown_and_record, daemon=True).start()

    def playback_and_review(self, wav_path, midi_notes, bpm, evaluation_results, note_rects, keyboard_width, note_height, roll_width, duration):
        """Play the recorded WAV and replay the MIDI visualization, showing which notes were correct/incorrect.

        This version updates a review label live as notes pass and provides an on-screen
        button to return to the home screen when the user is ready.
        """
        import time, sounddevice as sd, os
        from scipy.io.wavfile import read
        # optional OpenAI feedback
        try:
            import openai
            openai_available = True
        except Exception:
            openai_available = False
        # Load audio
        sr, data = read(wav_path)
        if data.ndim > 1:
            audio = data[:,0].astype(float) / 32768.0
        else:
            audio = data.astype(float) / 32768.0
        # Slightly increase playback volume (avoid clipping)
        gain = 1.6
        audio = np.clip(audio * gain, -1.0, 1.0)
        # Reset rectangles to blue
        for rect, st, et, y, note in note_rects:
            try:
                self.canvas.itemconfig(rect, fill='deepskyblue')
            except Exception:
                pass
        self.canvas.update()
        # Prepare a scrollable review area under the MIDI display (Text + Scrollbar)
        if not hasattr(self, 'review_container'):
            self.review_container = ttk.Frame(self.music_frame)
            self.review_container.pack(fill='x', padx=10, pady=6)
            # vertical scrollbar
            self.review_scroll = ttk.Scrollbar(self.review_container, orient='vertical')
            self.review_text = tk.Text(self.review_container, height=8, wrap='word', yscrollcommand=self.review_scroll.set, font=("Segoe UI", 11))
            self.review_scroll.config(command=self.review_text.yview)
            self.review_scroll.pack(side='right', fill='y')
            self.review_text.pack(side='left', fill='x', expand=True)
            self.review_text.config(state='disabled')
        # helper to update the review area safely
        def set_review(txt):
            try:
                self.review_text.config(state='normal')
                self.review_text.delete('1.0', 'end')
                self.review_text.insert('end', txt)
                self.review_text.see('end')
                self.review_text.config(state='disabled')
            except Exception:
                pass

        # live feedback history
        feedback_history = []
        announced = set()
        set_review('Playing back your recording and reviewing...')
        self.music_frame.update()
        # Play audio
        sd.play(audio, sr)
        start = time.time()
        steps = int(duration * 60)
        px_per_sec = roll_width / 4.0
        max_note = max((n for n,_,_ in midi_notes), default=72)
        for t in range(steps):
            cur_time = time.time() - start
            # update positions
            for rect, st, et, y, note in note_rects:
                x1 = keyboard_width + (st - cur_time) * px_per_sec
                x2 = keyboard_width + (et - cur_time) * px_per_sec
                if x2 < keyboard_width or x1 > keyboard_width + roll_width:
                    self.canvas.coords(rect, -10, -10, -10, -10)
                else:
                    self.canvas.coords(rect, x1, y+4, x2, y+note_height-4)
                # color according to evaluation_results when passed
                key = (note, st)
                if key in evaluation_results:
                    color = 'green' if evaluation_results[key] else 'red'
                    try:
                        self.canvas.itemconfig(rect, fill=color)
                    except Exception:
                        pass
            # live feedback: announce notes as they pass (once)
            for key, ok in list(evaluation_results.items()):
                note, st = key
                if st <= cur_time and key not in announced:
                    announced.add(key)
                    name = midi_to_note_name(note)
                    msg = (f"Good: {name} at {st:.2f}s" if ok else f"Missed: {name} at {st:.2f}s")
                    feedback_history.append(msg)
                    # keep last few entries visible
                    recent = "\n".join(feedback_history[-6:])
                    try:
                        set_review(recent)
                    except Exception:
                        pass
            self.canvas.update()
            time.sleep(1/60)
        sd.stop()
        # After playback, provide a single summarized feedback string below the live feed
        # show a short "compiling summary" message while we prepare the final summary
        try:
            set_review("Compiling summary...")
            self.music_frame.update()
        except Exception:
            pass

        wrong = [ (n, st) for (n, st), ok in evaluation_results.items() if not ok ]
        total_notes = max(1, len(note_rects))
        wrong_count = len(wrong)
        ratio = wrong_count / total_notes

        # Build a user-friendly summary that generalizes when many notes are missed
        if wrong_count == 0:
            summary = "Great job! All notes were played correctly in timing."
        elif ratio > 0.5:
            summary = "Many notes were played incorrectly — try slowing the piece down and isolating the difficult passages."
        elif ratio > 0.2:
            summary = "Some notes were played incorrectly. Focus on the highlighted measures and practice slowly."
        else:
            msgs = [f"{midi_to_note_name(n)} at {st:.2f}s" for n,st in wrong]
            summary = "You missed the following notes: " + ", ".join(msgs) + ". Try slowing down those passages and focus on hand coordination."

        # Optionally refine summary via OpenAI if available
        try:
            if openai_available and os.getenv('OPENAI_API_KEY'):
                openai.api_key = os.getenv('OPENAI_API_KEY')
                prompt = (
                    f"The performer made mistakes on these notes: {', '.join([f'{n}@{st:.2f}s' for n,st in wrong])}. "
                    "Provide concise, actionable, friendly feedback to help them improve, focusing on rhythm, hand coordination, and practice tips."
                )
                resp = openai.ChatCompletion.create(model="gpt-4", messages=[{"role":"user","content":prompt}], max_tokens=200)
                refined = resp.choices[0].message.content.strip()
                if refined:
                    summary = refined
        except Exception:
            pass

        # show final summary in the review label (append) then keep it until user returns home
        feedback_history.append("--- Summary ---")
        feedback_history.append(summary)
        try:
            set_review("\n".join(feedback_history[-8:]))
            self.music_frame.update()
        except Exception:
            pass

        # Do not auto-clear the summary; leave it visible until the user clicks Return to Home.
        # (previous behavior cleared the label after a timeout; removed so user can read at leisure)

        # Create an on-screen button so the user can return to the home screen when ready
        def return_home():
            try:
                self.canvas.delete('all')
            except Exception:
                pass
            # clean up review widgets
            try:
                self.review_container.destroy()
            except Exception:
                pass
            try:
                back_btn.destroy()
            except Exception:
                pass
            # show main UI
            self.music_frame.pack_forget()
            self.main_frame.pack(fill='both', expand=True)
            # update the feedback label on the main screen
            self.feedback_label.config(text=f"Recording complete! Saved to: {wav_path}\nFinal Accuracy: {100*sum(1 for ok in evaluation_results.values() if ok)/max(1,len(evaluation_results)):.1f}%")

        back_btn = ttk.Button(self.music_frame, text="Return to Home", command=return_home)
        # ensure the button is large enough and visible
        back_btn.pack(pady=8, ipadx=12, ipady=6)

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
