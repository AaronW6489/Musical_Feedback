import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from ui_style import setup_styles, create_title, create_section
import threading
from audio_utils import record_audio
from midi_utils import get_midi_total_beats, get_midi_notes
from midi_utils import get_midi_tempo
import librosa
import numpy as np
from analysis_utils import match_notes_to_audio, performance_summary, midi_to_freq
from tuner import open_tuner_window
from metronome import start_metronome

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
        # Start with a larger window that's a high fraction of the screen
        try:
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            # smaller default window: about 70% width and 65% height to avoid lots of empty space
            self.geometry(f"{int(sw*0.7)}x{int(sh*0.65)}")
        except Exception:
            self.geometry("900x600")
        self.configure(bg="#ecf0f1")

        setup_styles()

        # warm up pitch detection in background to avoid UI freeze later
        threading.Thread(target=self._warmup_pitch_detection, daemon=True).start()

        # Keep an internal instrument variable for prompts if needed, but remove the selection UI
        self.instrument = tk.StringVar(value="piano")
        self.sheet_path = None
        self.tempo_var = tk.IntVar(value=120)
        self.metronome_var = tk.BooleanVar(value=False)

        # create a root canvas with vertical scrollbar so pages can scroll vertically
        self.root_canvas = tk.Canvas(self, highlightthickness=0)
        self.root_v_scroll = ttk.Scrollbar(self, orient='vertical', command=self.root_canvas.yview)
        self.root_canvas.configure(yscrollcommand=self.root_v_scroll.set)
        # root canvas fills available area; vertical scrollbar sits on the right
        self.root_canvas.pack(fill='both', expand=True)
        self.root_v_scroll.pack(side='right', fill='y')

        # content frame placed inside the canvas
        self.content_frame = ttk.Frame(self.root_canvas)
        self._content_window = self.root_canvas.create_window((0,0), window=self.content_frame, anchor='nw')
        # ensure scrollregion updates and make content width follow canvas width
        self.content_frame.bind('<Configure>', lambda e: self.root_canvas.configure(scrollregion=self.root_canvas.bbox('all')))
        self.root_canvas.bind('<Configure>', lambda e: self.root_canvas.itemconfig(self._content_window, width=self.root_canvas.winfo_width()))

        self.main_frame = ttk.Frame(self.content_frame)
        self.music_frame = ttk.Frame(self.content_frame)

        self.canvas = None
        self.playhead = None

        # default visible window seconds for the piano-roll visualization
        self.visible_window_seconds = 6.0
        # pre-roll time so notes appear earlier (seconds)
        self.pre_roll_seconds = 2.0
        # enable mouse wheel to scroll the root canvas vertically (Windows)
        self.bind_all('<MouseWheel>', lambda e: self.root_canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units'))

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
        # Pages container: we'll show one page at a time
        self.pages_frame = ttk.Frame(self.content_frame, style='App.TFrame')
        self.pages_frame.pack(fill='both', expand=True, padx=12, pady=12)

        # main_frame becomes the Home page
        self.main_frame = ttk.Frame(self.pages_frame, style='Card.TFrame', padding=12)
        self.main_frame.pack(fill='both', expand=True)
        create_title(self.main_frame, "Instrument Performance Analyzer")
        # NOTE: removed instrument selection UI to simplify the flow (instrument kept internally as 'piano' by default)
        # Short instruction on the home page
        instr = ttk.Label(self.main_frame, text="Upload your sheet music, select your tempo, and begin recording", font=("Segoe UI", 11))
        instr.pack(pady=(6,12))

        # Pre-play section (previously 'Sheet Music')
        sheet_frame = create_section(self.main_frame, "Pre-play")
        sheet_frame.pack(fill="x", pady=10)
        ttk.Button(sheet_frame, text="Upload Sheet Music (MIDI/Image)", command=self.upload_sheet).pack(pady=5)
        self.sheet_label = ttk.Label(sheet_frame, text="No file uploaded.")
        self.sheet_label.pack()

        # Recording section (minimal UI)
        record_frame = create_section(self.main_frame, "Recording")
        record_frame.pack(fill="x", pady=10)
        ttk.Button(record_frame, text="Start Recording", command=self.enter_selection_mode).pack(pady=5)
        # Download last recording button
        btn_row = ttk.Frame(record_frame)
        self.download_btn = ttk.Button(btn_row, text="Download Last Recording", command=self.download_recording)
        self.download_btn.pack(side='left', padx=6)
        btn_row.pack(pady=6)

        # Feedback section below Recording
        feedback_frame = create_section(self.main_frame, "Feedback")
        feedback_frame.pack(fill='x', pady=6)
        # use a plain Label for the feedback area so it looks like normal text instead of a textbox
        self.feedback_label = ttk.Label(feedback_frame, text= 'Feedback will appear here after review', wraplength=900, justify='left')
        self.feedback_label.pack(fill='x', padx=6, pady=8)

        # Move tempo/metronome/tuner controls to the Pre-play section
        tempo_row = ttk.Frame(sheet_frame)
        ttk.Label(tempo_row, text="Tempo (BPM):").pack(side='left', padx=(0,6))
        self.tempo_spin = ttk.Spinbox(tempo_row, from_=20, to=300, textvariable=self.tempo_var, width=6)
        self.tempo_spin.pack(side='left')
        ttk.Checkbutton(tempo_row, text="Enable Metronome", variable=self.metronome_var).pack(side='left', padx=10)
        tempo_row.pack(pady=4)
        btn_row2 = ttk.Frame(sheet_frame)
        ttk.Button(btn_row2, text="Tuner", command=self.open_tuner).pack(side='left', padx=6)
        btn_row2.pack(pady=6)

        # flag whether user downloaded the last recording; used to control saved message
        self.downloaded_last = False

        # Music frame for piano roll display (responsive) - will be shown as the Recording page
        self.canvas_height = 600
        # create horizontal scrollbar but don't show it until selection mode
        self.h_scroll = ttk.Scrollbar(self.music_frame, orient='horizontal')
        self.canvas = tk.Canvas(self.music_frame, bg='#1e272e', highlightthickness=0, xscrollcommand=self.h_scroll.set)
        self.canvas.pack(fill='both', expand=True, pady=10)
        self.h_scroll.config(command=self.canvas.xview)
        self.music_status = ttk.Label(self.music_frame, text="", font=("Segoe UI", 12))
        self.music_status.pack()
        # selection state (seconds)
        self.selection_start = None
        self.selection_end = None
        # visual selection markers
        self.selection_start_line = None
        self.selection_end_line = None
        self.selection_label = ttk.Label(self.music_frame, text="Selection: full piece")
        self.selection_label.pack(pady=4)

        # Canvas selection handlers (set start/end by left clicks, right click to clear)
        def canvas_click(event):
            # map x to seconds using canvas coordinates (respect scrolling)
            if not hasattr(self, 'px_per_sec_full') or not hasattr(self, 'full_duration'):
                return
            # convert event x to canvas coordinate (accounts for scroll)
            x = self.canvas.canvasx(event.x)
            # clamp inside roll area
            kb = getattr(self, 'keyboard_width', 120)
            if x < kb:
                x = kb
            rel = x - kb
            t = rel / self.px_per_sec_full
            t = max(0.0, min(self.full_duration, t))
            if self.selection_start is None:
                self.selection_start = t
                # draw start line
                try:
                    if self.selection_start_line:
                        self.canvas.delete(self.selection_start_line)
                except Exception:
                    pass
                sx = kb + self.selection_start * self.px_per_sec_full
                self.selection_start_line = self.canvas.create_line(sx, 0, sx, self.canvas.winfo_height(), fill='yellow', width=2)
                # update hint: ask user to select end next
                try:
                    if hasattr(self, 'selection_hint'):
                        self.selection_hint.config(text="Click on where you would like to end (right click to reset)")
                except Exception:
                    pass
            else:
                # Ensure the end cannot be before the start; clamp to start if user clicks earlier
                try:
                    if t < self.selection_start:
                        t = self.selection_start
                except Exception:
                    pass
                self.selection_end = t
                try:
                    if self.selection_end_line:
                        self.canvas.delete(self.selection_end_line)
                except Exception:
                    pass
                ex = kb + self.selection_end * self.px_per_sec_full
                self.selection_end_line = self.canvas.create_line(ex, 0, ex, self.canvas.winfo_height(), fill='orange', width=2)
                # once both start and end are selected, hide the hint
                try:
                    if hasattr(self, 'selection_hint'):
                        self.selection_hint.config(text="")
                        self.selection_hint.pack_forget()
                except Exception:
                    pass
            # update label
            s = self.selection_start
            e = self.selection_end if self.selection_end is not None else s
            self.selection_label.config(text=f"Selection: {s:.2f}s - {e:.2f}s")

        def canvas_right_click(event):
            # clear selection
            self.selection_start = None
            self.selection_end = None
            try:
                if self.selection_start_line:
                    self.canvas.delete(self.selection_start_line)
                if self.selection_end_line:
                    self.canvas.delete(self.selection_end_line)
            except Exception:
                pass
            self.selection_start_line = None
            self.selection_end_line = None
            self.selection_label.config(text="Selection: full piece")
            # reset instructional hint to start instruction
            try:
                if hasattr(self, 'selection_hint'):
                    self.selection_hint.config(text="Click on where you would like to start (right click to reset)")
                    # ensure it's visible
                    try:
                        self.selection_hint.pack(pady=4)
                    except Exception:
                        pass
            except Exception:
                pass
            # reset instructional hints
            try:
                if hasattr(self, 'selection_hint'):
                    self.selection_hint.config(text="Click on where you would like to start (right click to reset)")
            except Exception:
                pass

        # bind events (inside create_widgets scope)
        self.canvas.bind('<Button-1>', canvas_click)
        self.canvas.bind('<Button-3>', canvas_right_click)

        # Create a simple bottom navigation bar
        # Create a fixed bottom navigation bar attached to the root window (so it doesn't scroll)
        try:
            nav = ttk.Frame(self, style='BottomNav.TFrame')
            nav.pack(side='bottom', fill='x')
            self.nav_home_btn = ttk.Button(nav, text='Home', style='Nav.TButton', command=lambda: self.show_page('home'))
            self.nav_home_btn.pack(side='left', expand=True, fill='x')
            self.nav_tuner_btn = ttk.Button(nav, text='Tuner', style='Nav.TButton', command=lambda: self.show_page('tuner'))
            self.nav_tuner_btn.pack(side='left', expand=True, fill='x')
            self.nav_record_btn = ttk.Button(nav, text='Recording', style='Nav.TButton', command=lambda: self.show_page('recording'))
            self.nav_record_btn.pack(side='left', expand=True, fill='x')
            # Recording should be disabled until a sheet is uploaded
            try:
                self.nav_record_btn.state(['disabled'])
            except Exception:
                pass
        except Exception:
            pass
        # ensure tuner and recording pages exist as frames so show_page can switch
        if not hasattr(self, 'tuner_frame'):
            self.tuner_frame = ttk.Frame(self.pages_frame, style='Card.TFrame', padding=12)
            ttk.Label(self.tuner_frame, text='Tuner', font=("Segoe UI", 20)).pack(pady=8)
            # Tuner UI elements integrated into the tuner page
            self.tuner_pitch_label = ttk.Label(self.tuner_frame, text='-- Hz / --', font=("Segoe UI", 18))
            self.tuner_pitch_label.pack(pady=8)
            btns = ttk.Frame(self.tuner_frame)
            ttk.Button(btns, text='Start Tuner', command=self._start_tuner).pack(side='left', padx=6)
            ttk.Button(btns, text='Stop Tuner', command=lambda: setattr(self, 'tuner_stop', True)).pack(side='left', padx=6)
            ttk.Button(btns, text='Back', command=lambda: self.show_page('home')).pack(side='left', padx=6)
            btns.pack(pady=6)
        # the existing music_frame will act as the recording page
        self.recording_frame = self.music_frame

        # helper to switch pages
        def _scroll_frame_to_top(frame):
            try:
                # ensure geometry is up-to-date
                self.root_canvas.update_idletasks()
                # compute frame y relative to content_frame
                frame_y = frame.winfo_rooty() - self.content_frame.winfo_rooty()
                content_h = max(1, self.content_frame.winfo_height())
                canvas_h = max(1, self.root_canvas.winfo_height())
                denom = max(1, content_h - canvas_h)
                frac = min(1.0, max(0.0, frame_y / denom)) if denom > 0 else 0.0
                self.root_canvas.yview_moveto(frac)
            except Exception:
                try:
                    self.root_canvas.yview_moveto(0.0)
                except Exception:
                    pass

        def show_page(name):
            # hide all known pages first
            for f in (self.main_frame, self.tuner_frame, self.recording_frame):
                try:
                    f.pack_forget()
                except Exception:
                    pass

            # pack the requested page
            try:
                if name == 'home':
                    self.main_frame.pack(fill='both', expand=True)
                    _scroll_frame_to_top(self.main_frame)
                elif name == 'tuner':
                    self.tuner_frame.pack(fill='both', expand=True)
                    _scroll_frame_to_top(self.tuner_frame)
                elif name == 'recording':
                    self.recording_frame.pack(fill='both', expand=True)
                    # reset piano-roll horizontal scroll so keyboard is visible immediately
                    try:
                        if hasattr(self, 'canvas') and self.canvas is not None:
                            self.canvas.xview_moveto(0.0)
                    except Exception:
                        pass
                    _scroll_frame_to_top(self.recording_frame)
                else:
                    self.main_frame.pack(fill='both', expand=True)
                    _scroll_frame_to_top(self.main_frame)
            except Exception:
                pass
        # attach to self for use elsewhere
        self.show_page = show_page

        # show home by default
        try:
            self.show_page('home')
        except Exception:
            pass

    def upload_sheet(self):
        filetypes = [("MIDI files", "*.mid *.midi"), ("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select Sheet Music", filetypes=filetypes)
        if path:
            self.sheet_path = path
            self.sheet_label.config(text=f"Uploaded: {path.split('/')[-1]}")
            # try to prefill tempo from MIDI
            try:
                tempo_bpm = get_midi_tempo(path)
                self.tempo_var.set(tempo_bpm)
            except Exception:
                pass
            # do not auto-enable recording nav here; enable only when user starts recording
        else:
            self.sheet_label.config(text="No file uploaded.")

    def start_recording(self):
        if not self.sheet_path:
            return
        if not self.sheet_path.lower().endswith(('.mid', '.midi')):
            messagebox.showwarning("MIDI Required", "Live feedback currently requires a MIDI file.")
            return

        # If the Begin Recording button is still present, don't accidentally start a full-piece recording.
        # This prevents the previous UX where Start Recording could be triggered without the user
        # pressing Begin Recording after making a selection.
        try:
            if hasattr(self, 'begin_btn') and self.begin_btn is not None and getattr(self.begin_btn, 'winfo_exists', lambda: False)():
                messagebox.showinfo("Selection Pending", "Please click 'Begin Recording' to start your selected region, or right-click on the piano roll to clear selection.")
                return
        except Exception:
            pass

        # tempo and durations
        bpm = int(self.tempo_var.get() or 120)
        total_beats = get_midi_total_beats(self.sheet_path)
        full_duration = total_beats * 60 / bpm

        # store for selection mapping (canvas click handlers use these)
        self.full_duration = full_duration
        self.keyboard_width = 120
        roll_width = 760
        self.px_per_sec_full = roll_width / max(1.0, full_duration)

        # If the user selected a region, use that selection
        if self.selection_start is not None and self.selection_end is not None:
            # clamp
            s = max(0.0, min(self.selection_start, self.selection_end))
            e = min(full_duration, max(self.selection_start, self.selection_end))
            duration = max(0.5, e - s)
            selection_offset = s
        else:
            duration = full_duration
            selection_offset = 0.0
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
        # reset horizontal scroll so keyboard (left) is visible when recording begins
        try:
            # move view to the leftmost position
            try:
                self.canvas.xview_moveto(0.0)
            except Exception:
                pass
            # hide horizontal scrollbar until selection mode requires it
            try:
                self.h_scroll.pack_forget()
            except Exception:
                pass
        except Exception:
            pass
        self.music_status.config(text="Get ready! Recording starts soon...")
        self.update()

        def countdown_and_record():
            import time
            import sounddevice as sd
            from scipy.io.wavfile import write, read
            import tempfile
            import threading as _threading
            chunk_sec = 0.5
            fs = 44100
            for i in range(5, 0, -1):
                self.music_status.config(text=f"Recording starts in {i}...")
                self.update()
                time.sleep(1)
            self.music_status.config(text="Recording!")
            self.update()
            # Optional metronome thread container (actual start will occur when recording clock starts)
            metronome_stop = _threading.Event()
            metronome_thread = None
            self.draw_keyboard(min_note, max_note, note_height, keyboard_width, height)
            playhead = self.canvas.create_line(keyboard_width, 0, keyboard_width, height, fill='red', width=3)
            # Visible window in seconds (how many seconds fit the roll_width on screen)
            visible_window = 4.0
            # px_per_sec maps seconds to pixels for on-screen movement
            px_per_sec = (roll_width) / max(0.1, visible_window)
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
            # Use wall-clock timing to keep visuals aligned with real tempo and avoid drift
            record_start_time = time.time()
            # Start metronome aligned to the piece beat grid if requested.
            try:
                if self.metronome_var.get():
                    import math
                    interval = 60.0 / float(max(1.0, bpm))
                    # compute time from selection_offset to next beat boundary
                    try:
                        sel_off = float(selection_offset)
                    except Exception:
                        sel_off = 0.0
                    next_beat = math.ceil(sel_off / interval) * interval
                    offset_sec = max(0.0, next_beat - sel_off)
                    start_time_met = record_start_time + offset_sec
                    try:
                        metronome_thread = start_metronome(metronome_stop, bpm, fs=fs, start_time=start_time_met, duration=duration)
                    except Exception:
                        metronome_thread = None
            except Exception:
                pass
            # include pre-roll so visuals can start earlier than audio
            pre_roll = getattr(self, 'pre_roll_seconds', 0.0)
            end_time_abs = record_start_time + duration + pre_roll
            frame_interval = 1/60
            while time.time() < end_time_abs:
                # cur_time is audio timeline seconds relative to selection_offset, but we subtract pre_roll
                # so visuals start earlier (notes appear before the audio plays)
                cur_time = selection_offset - pre_roll + (time.time() - record_start_time)
                # Move notes
                for rect, start_time, end_time, y, note in note_rects:
                    x1 = keyboard_width + (start_time - cur_time) * px_per_sec
                    x2 = keyboard_width + (end_time - cur_time) * px_per_sec
                    if x2 < keyboard_width or x1 > keyboard_width + roll_width:
                        self.canvas.coords(rect, -10, -10, -10, -10)
                    else:
                        self.canvas.coords(rect, x1, y+4, x2, y+note_height-4)
                # Real-time analysis every chunk_sec
                # compute the elapsed seconds since start and run analysis at chunk boundaries
                elapsed = time.time() - record_start_time
                if int(elapsed / chunk_sec) != int((elapsed - frame_interval) / chunk_sec) and cur_sample > chunk_samples:
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
                time.sleep(frame_interval)
            stream.stop()
            # stop metronome if running
            try:
                metronome_stop.set()
            except Exception:
                pass
            # Save audio
            temp_dir = tempfile.gettempdir()
            wav_path = temp_dir + "/user_recording.wav"
            write(wav_path, fs, audio_buffer)
            self.music_status.config(text=f"Recording complete!\nFinal Accuracy: {100*correct_count/max(1,total_count):.1f}%")
            # After recording, run playback review: replay the recorded audio and the MIDI with colored feedback
            try:
                # pass selection_offset and selection duration so review can limit accuracy to selected notes
                self.playback_and_review(wav_path, midi_notes, bpm, evaluation_results, note_rects, keyboard_width, note_height, roll_width, duration, selection_offset=selection_offset, selection_duration=duration)
            except Exception:
                pass

        threading.Thread(target=countdown_and_record, daemon=True).start()

    def enter_selection_mode(self):
        """Allow user to pick start/end on the piano roll before actually recording.
        Shows a Begin Recording button that will call start_recording_confirmed.
        """
        # make sure a MIDI is loaded
        if not self.sheet_path or not self.sheet_path.lower().endswith(('.mid', '.midi')):
            messagebox.showwarning("MIDI Required", "Please upload a MIDI file first to use selection.")
            return
        # draw a windowed piano roll so user can click to set selection; enable horizontal scroll
        total_beats = get_midi_total_beats(self.sheet_path)
        bpm = int(self.tempo_var.get() or 120)
        midi_notes = get_midi_notes(self.sheet_path)
        # show music frame and ensure note rectangles are present
        self.main_frame.pack_forget()
        self.music_frame.pack(fill='both', expand=True)
        # make visible window width relative to the current window size (clamp)
        visible_window_px = min(1000, max(600, int(self.winfo_width() * 0.85)))
        visible_window_seconds = 6.0  # show ~6 seconds window by default
        self.canvas.config(width=visible_window_px)
        self.canvas.delete('all')

        # prepare basic mapping vars
        full_duration = total_beats * 60 / bpm
        self.full_duration = full_duration
        self.keyboard_width = 120
        # virtual full roll width in pixels (for scrollregion)
        virtual_roll_width = max(int(full_duration * (visible_window_px / visible_window_seconds)), visible_window_px)
        self.px_per_sec_full = (virtual_roll_width - self.keyboard_width) / max(1.0, full_duration)
        roll_width = visible_window_px

        # draw keyboard and notes
        min_note = min((n for n,_,_ in midi_notes), default=21)
        max_note = max((n for n,_,_ in midi_notes), default=108)
        note_range = max_note - min_note + 1
        note_height = max(18, min(40, 600 // note_range))
        height = note_height * note_range
        # Configure the canvas size and scrollregion for the virtual full piece
        self.canvas.config(height=height, scrollregion=(0,0, self.keyboard_width + virtual_roll_width, height))
        # show horizontal scrollbar for selection
        try:
            self.h_scroll.pack(side='bottom', fill='x')
        except Exception:
            pass
        # align initial view to the start
        try:
            self.canvas.xview_moveto(0.0)
        except Exception:
            pass
        self.draw_keyboard(min_note, max_note, note_height, self.keyboard_width, height)
        # draw notes for selection
        for note, start_beat, duration_beat in midi_notes:
            start_time = start_beat * (60.0 / bpm)
            end_time = (start_beat + duration_beat) * (60.0 / bpm)
            y = (max_note - note) * note_height
            x1 = self.keyboard_width + start_time * self.px_per_sec_full
            x2 = self.keyboard_width + end_time * self.px_per_sec_full
            self.canvas.create_rectangle(x1, y+4, x2, y+note_height-4, fill='deepskyblue', outline='black', tags='note')
        # add Begin Recording button
        if hasattr(self, 'begin_btn') and self.begin_btn.winfo_exists():
            return
        self.begin_btn = ttk.Button(self.music_frame, text='Begin Recording', command=lambda: self.start_recording_confirmed(midi_notes))
        self.begin_btn.pack(pady=6)
        # Add selection hint label
        try:
            if not hasattr(self, 'selection_hint'):
                self.selection_hint = ttk.Label(self.music_frame, text="Click on where you would like to start (right click to reset)")
                self.selection_hint.pack(pady=4)
            else:
                self.selection_hint.config(text="Click on where you would like to start (right click to reset)")
        except Exception:
            pass
        # Enable access to the Recording navigation tab since user is actively preparing to record
        try:
            if hasattr(self, 'nav_record_btn'):
                try:
                    self.nav_record_btn.state(['!disabled'])
                except Exception:
                    try:
                        self.nav_record_btn.config(state='normal')
                    except Exception:
                        pass
        except Exception:
            pass
        # show the recording page and ensure it's scrolled into view
        try:
            self.show_page('recording')
        except Exception:
            pass

    def start_recording_confirmed(self, midi_notes):
        # remove begin button and call original start_recording flow (it will re-read midi_notes)
        try:
            if hasattr(self, 'begin_btn'):
                self.begin_btn.destroy()
        except Exception:
            pass
        # ensure selection values remain available; call start_recording which expects sheet_path set
        self.start_recording()

    def open_tuner(self):
        # Show the in-app tuner page
        try:
            # ensure stop flag is clear
            self.tuner_stop = False
            # switch to tuner page
            try:
                self.show_page('tuner')
            except Exception:
                pass
        except Exception:
            messagebox.showwarning("Tuner Error", "Unable to open tuner on this system.")

    def _close_tuner(self):
        try:
            self.tuner_stop = True
            if hasattr(self, 'tuner_frame'):
                self.tuner_frame.pack_forget()
        except Exception:
            pass
        # return to main screen
        self.main_frame.pack(fill='both', expand=True)

    def _tuner_loop(self):
        # lightweight tuner loop using sounddevice + librosa piptrack
        try:
            import time
            import sounddevice as sd
            fs = 22050
            block = 2048
            def callback(indata, frames, time_info, status):
                if getattr(self, 'tuner_stop', False):
                    raise sd.CallbackStop()
                y = indata[:,0].astype(float) / 32768.0
                try:
                    pitches, mags = librosa.piptrack(y=y, sr=fs, fmin=50, fmax=2000)
                    idx = np.argmax(mags)
                    i = idx // mags.shape[1]
                    j = idx % mags.shape[1]
                    freq = pitches[i,j] if mags[i,j] > 0 else None
                except Exception:
                    freq = None
                try:
                    if freq:
                        # convert freq to nearest note name
                        try:
                            midi = int(round(69 + 12 * np.log2(freq / 440.0)))
                            note_name = midi_to_note_name(midi)
                        except Exception:
                            note_name = f"{freq:.1f} Hz"
                        self.tuner_pitch_label.config(text=note_name)
                    else:
                        self.tuner_pitch_label.config(text='--')
                except Exception:
                    pass
            with sd.InputStream(samplerate=fs, channels=1, blocksize=block, callback=callback):
                while not getattr(self, 'tuner_stop', False):
                    time.sleep(0.1)
        except Exception:
            pass

    def _start_tuner(self):
        try:
            # reset stop flag and start tuner thread; safe to call multiple times
            self.tuner_stop = False
            threading.Thread(target=self._tuner_loop, daemon=True).start()
        except Exception:
            pass

    def download_recording(self):
        # Offer the last temp recording for save-as
        import tempfile, shutil
        temp_dir = tempfile.gettempdir()
        src = temp_dir + "/user_recording.wav"
        try:
            path = filedialog.asksaveasfilename(defaultextension='.wav', filetypes=[('WAV files','*.wav')])
            if path:
                shutil.copy(src, path)
                # mark that user downloaded the last recording so home shows saved path
                self.downloaded_last = True
        except Exception:
            messagebox.showwarning("No recording", "No recording available to download.")

    def playback_and_review(self, wav_path, midi_notes, bpm, evaluation_results, note_rects, keyboard_width, note_height, roll_width, duration, selection_offset=0.0, selection_duration=None):
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

        # live feedback history (persistent for this session until Return to Home)
        feedback_history = []
        announced = set()
        set_review('Playing back your recording and reviewing...')
        self.music_frame.update()
        # Ensure only one playback control exists at a time
        if hasattr(self, 'playback_controls') and self.playback_controls.winfo_exists():
            # reuse existing controls (reset values)
            self.playback_controls.destroy()
        self.playback_controls = ttk.Frame(self.music_frame)
        self.playback_controls.pack(fill='x', pady=6)

        is_playing = {'val': True}
        # playback state: track where playback started, what audio position that corresponds to, and speed
        play_state = {'start_time': None, 'start_pos': 0.0, 'speed': 1.0, 'paused_at': 0.0}

        # speed and volume controls
        speed_var = tk.DoubleVar(value=1.0)
        volume_var = tk.DoubleVar(value=1.0)

        # progress bar
        progress = ttk.Scale(self.playback_controls, from_=0, to=duration, orient='horizontal')
        progress.pack(side='right', fill='x', expand=True, padx=8)
        user_seeking = {'val': False}

        def on_seek_start(event=None):
            # user started interacting; pause playback
            user_seeking['val'] = True
            if is_playing['val']:
                # pause and record paused_at (account for speed)
                sd.stop()
                if play_state.get('start_time') is not None:
                    cur_pos = play_state['start_pos'] + (time.time() - play_state['start_time']) * play_state.get('speed', 1.0)
                else:
                    cur_pos = play_state.get('paused_at', 0.0)
                play_state['paused_at'] = cur_pos
                is_playing['val'] = False
                play_pause_btn.config(text='Play')

        def on_seek_end(event=None):
            user_seeking['val'] = False
            # set paused_at to progress value
            try:
                play_state['paused_at'] = float(progress.get())
            except Exception:
                pass

        def _restart_playback_from(pos_seconds):
            """Stop current playback and start from pos_seconds using current speed/volume."""
            try:
                sd.stop()
            except Exception:
                pass
            try:
                start_sample = int(max(0, int(pos_seconds * sr)))
                seg = np.clip(audio[start_sample:], -1.0, 1.0) * float(volume_var.get())
                speed = float(speed_var.get())
                play_sr = max(1, int(sr * speed))
                sd.play(seg, play_sr)
                play_state['start_time'] = time.time()
                play_state['start_pos'] = pos_seconds
                play_state['speed'] = speed
                play_state['paused_at'] = pos_seconds
                is_playing['val'] = True
                play_pause_btn.config(text='Pause')
            except Exception:
                pass

        # debounce timers to prevent excessive restarts during slider drags
        _debounce_speed_id = None
        _debounce_volume_id = None

        def _debounced_restart():
            try:
                if is_playing['val']:
                    if play_state.get('start_time') is not None:
                        cur_pos = play_state['start_pos'] + (time.time() - play_state['start_time']) * play_state.get('speed', 1.0)
                    else:
                        cur_pos = play_state.get('paused_at', 0.0)
                    _restart_playback_from(cur_pos)
            except Exception:
                pass

        def on_speed_change(event=None):
            nonlocal _debounce_speed_id
            try:
                # cancel previous scheduled restart
                if _debounce_speed_id:
                    try:
                        self.after_cancel(_debounce_speed_id)
                    except Exception:
                        pass
                # schedule restart after short idle
                _debounce_speed_id = self.after(200, _debounced_restart)
            except Exception:
                pass

        def on_volume_change(event=None):
            nonlocal _debounce_volume_id
            try:
                if _debounce_volume_id:
                    try:
                        self.after_cancel(_debounce_volume_id)
                    except Exception:
                        pass
                _debounce_volume_id = self.after(200, _debounced_restart)
            except Exception:
                pass

        progress.bind('<Button-1>', on_seek_start)
        progress.bind('<ButtonRelease-1>', on_seek_end)

        # play/pause handling uses sounddevice.play which is global; implement pause by stopping and resuming from position
        def toggle_play():
            if is_playing['val']:
                # pause
                try:
                    # compute logical current position first
                    if play_state.get('start_time') is not None:
                        cur_pos = play_state['start_pos'] + (time.time() - play_state['start_time']) * play_state.get('speed', 1.0)
                    else:
                        cur_pos = play_state.get('paused_at', 0.0)
                    play_state['paused_at'] = cur_pos
                except Exception:
                    play_state['paused_at'] = play_state.get('paused_at', 0.0)
                try:
                    sd.stop()
                except Exception:
                    pass
                is_playing['val'] = False
                play_pause_btn.config(text='Play')
            else:
                # resume from paused_at
                start_pos = play_state.get('paused_at', 0.0)
                _restart_playback_from(start_pos)

        play_pause_btn = ttk.Button(self.playback_controls, text='Pause', command=toggle_play)
        play_pause_btn.pack(side='left', padx=6)
        ttk.Label(self.playback_controls, text='Speed:').pack(side='left', padx=(10,4))
        # wider speed range for more noticeable changes
        speed_scale = ttk.Scale(self.playback_controls, from_=0.5, to=3.0, variable=speed_var, orient='horizontal', length=140)
        speed_scale.pack(side='left')
        ttk.Label(self.playback_controls, text='Volume:').pack(side='left', padx=(8,4))
        volume_scale = ttk.Scale(self.playback_controls, from_=0.0, to=2.0, variable=volume_var, orient='horizontal', length=120)
        volume_scale.pack(side='left')
        # bind change handlers so speed/volume changes take effect immediately
        speed_scale.bind('<ButtonRelease-1>', on_speed_change)
        speed_scale.bind('<B1-Motion>', on_speed_change)
        volume_scale.bind('<ButtonRelease-1>', on_volume_change)
        volume_scale.bind('<B1-Motion>', on_volume_change)

        # Before starting playback, run per-note analysis so we can reference duration/volume/pitch during playback
        try:
            sr2, y2 = read(wav_path)
            if y2.ndim > 1:
                y_mono = y2[:,0].astype(float) / 32768.0
            else:
                y_mono = y2.astype(float) / 32768.0
            full_analysis = match_notes_to_audio((y_mono*32768).astype('int16'), sr2, midi_notes, bpm)
            # Build a lookup by (note, start_time) to the analysis tuple for quick access
            analysis_map = { (note, round(st,3)): (note, detected_freq, correct, (st,et), dur_est, rms)
                            for (note, detected_freq, correct, (st,et), dur_est, rms) in full_analysis }
        except Exception:
            full_analysis = []
            analysis_map = {}

        # start playing audio (from beginning) using current speed and volume
        try:
            vol0 = float(volume_var.get())
            speed0 = float(speed_var.get())
            seg0 = np.clip(audio * vol0, -1.0, 1.0)
            play_sr0 = max(1, int(sr * speed0))
            sd.play(seg0, play_sr0)
            play_state['start_time'] = time.time()
            play_state['start_pos'] = 0.0
            play_state['speed'] = speed0
            is_playing['val'] = True
        except Exception:
            pass
        px_per_sec = roll_width / 4.0
        max_note = max((n for n,_,_ in midi_notes), default=72)
        # use play_state to compute current playback time so pause/resume and seeking are reflected
        while True:
            if is_playing['val'] and play_state.get('start_time'):
                # Logical current time accounts for where playback started and the current speed multiplier
                elapsed = time.time() - play_state['start_time']
                cur_time = play_state.get('start_pos', 0.0) + elapsed * play_state.get('speed', 1.0)
            else:
                cur_time = play_state.get('paused_at', 0.0)

            if cur_time >= duration:
                break

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
                    # Only record incorrect notes in the persistent log
                    if not ok:
                        # Compose a persistent, informative sentence using analysis_map when available
                        lookup_key = (note, round(st,3))
                        analysis = analysis_map.get(lookup_key)
                        if analysis:
                            _, detected_freq, correct_flag, (ast,aet), dur_est, rms = analysis
                            # Determine primary reason (pitch/duration/volume)
                            reason = None
                            # Pitch check
                            if detected_freq:
                                expected_freq = midi_to_freq(note)
                                cents = 1200 * np.log2(detected_freq / expected_freq) if expected_freq>0 else 0
                                if abs(cents) > 30:
                                    reason = f"wrong note (about {cents:+.0f} cents off)"
                                elif abs(cents) > 8:
                                    reason = f"slightly out of tune ({cents:+.0f} cents)"
                                else:
                                    reason = "wrong note"
                            else:
                                reason = "wrong note"

                            # Duration check (if more severe than pitch)
                            expected_dur = aet - ast
                            if dur_est > expected_dur * 1.25:
                                mag = 'much' if dur_est > expected_dur * 1.5 else 'a little'
                                reason = f"too long ({mag})"
                            elif dur_est < expected_dur * 0.7:
                                mag = 'much' if dur_est < expected_dur * 0.5 else 'a little'
                                reason = f"too short ({mag})"

                            # Volume check (if more severe than previous)
                            loud_thresh = 0.04
                            if rms > loud_thresh * 1.8:
                                reason = "too loud"
                            elif rms < loud_thresh * 0.6:
                                reason = "too quiet"

                            sentence = f"{name} at {st:.2f}s: {reason}."
                        else:
                            sentence = f"{name} at {st:.2f}s: wrong note."
                        feedback_history.append(sentence)
                        # update the review area to show accumulated log (do not clear on each note)
                        try:
                            set_review("\n".join(feedback_history[-50:]))
                        except Exception:
                            pass

            self.canvas.update()
            # update progress scale when not seeking
            try:
                if not user_seeking['val']:
                    progress.set(min(duration, cur_time))
            except Exception:
                pass

            time.sleep(1/60)
        # end playback loop
        sd.stop()
        # After playback, provide a single summarized feedback string below the live feed
        # show a short "compiling summary" message while we prepare the final summary
        try:
            set_review("Compiling summary...")
            self.music_frame.update()
        except Exception:
            pass

        # Determine which MIDI notes fall inside the selected window for accuracy calculation
        sel_offset = float(selection_offset or 0.0)
        sel_dur = float(selection_duration) if selection_duration is not None else duration
        sel_end = sel_offset + sel_dur
        # consider notes whose start time is within [sel_offset, sel_end)
        selected_notes = []
        for note, start_beat, duration_beat in midi_notes:
            st = start_beat * 60 / bpm
            if st >= sel_offset and st < sel_end:
                selected_notes.append((note, st))

        wrong = [ (n, st) for (n, st), ok in evaluation_results.items() if not ok and (n, st) in selected_notes ]
        total_notes = max(1, len(selected_notes))
        wrong_count = len(wrong)
        ratio = wrong_count / total_notes if total_notes else 0.0

        # Enrich feedback with duration and volume analysis using recorded audio
        try:
            # read audio again to analyze per-note features
            sr2, y2 = read(wav_path)
            if y2.ndim > 1:
                y_mono = y2[:,0].astype(float) / 32768.0
            else:
                y_mono = y2.astype(float) / 32768.0
            analysis_results = match_notes_to_audio((y_mono*32768).astype('int16'), sr2, midi_notes, bpm)
            # analysis_results items: (note, detected_freq, correct, (start,end), dur_est, rms)
        except Exception:
            analysis_results = []

        # Build a more informative summary derived from the per-note analysis (pitch/duration/volume)
        try:
            pitch_issues = []
            dur_issues = []
            vol_issues = []
            other = []
            for ar in analysis_results:
                note, detected_freq, correct, (st,et), dur_est, rms = ar
                if correct:
                    continue
                note_name = midi_to_note_name(note)
                issue_parts = []
                # pitch
                if detected_freq:
                    expected_freq = midi_to_freq(note)
                    cents = 1200 * np.log2(detected_freq / expected_freq) if expected_freq>0 else 0
                    if abs(cents) > 30:
                        issue_parts.append('out of tune')
                        pitch_issues.append((note_name, st, cents))
                    elif abs(cents) > 8:
                        issue_parts.append('slightly out of tune')
                        pitch_issues.append((note_name, st, cents))
                else:
                    issue_parts.append('missed pitch')
                    pitch_issues.append((note_name, st, None))
                # duration
                expected_dur = et - st
                if dur_est > expected_dur * 1.25:
                    dur_issues.append((note_name, st, dur_est - expected_dur))
                    issue_parts.append('held too long')
                elif dur_est < expected_dur * 0.7:
                    dur_issues.append((note_name, st, expected_dur - dur_est))
                    issue_parts.append('cut short')
                # volume
                loud_thresh = 0.04
                if rms > loud_thresh * 1.8:
                    vol_issues.append((note_name, st, rms))
                    issue_parts.append('too loud')
                elif rms < loud_thresh * 0.6:
                    vol_issues.append((note_name, st, rms))
                    issue_parts.append('too quiet')
                if not issue_parts:
                    other.append((note_name, st))

            if wrong_count == 0:
                summary = "Great job! All notes were played correctly in timing."
            else:
                pct = 100.0 * wrong_count / max(1, total_notes)
                top_issue = None
                # choose the most common issue category
                if len(pitch_issues) >= max(len(dur_issues), len(vol_issues)):
                    top_issue = 'pitch'
                elif len(dur_issues) >= max(len(pitch_issues), len(vol_issues)):
                    top_issue = 'duration'
                else:
                    top_issue = 'volume'

                summary_lines = [f"{wrong_count}/{total_notes} notes had issues ({pct:.0f}%)."]
                if top_issue == 'pitch':
                    summary_lines.append("Most problems were pitch-related — try slow pitch-focused practice and tuning checks.")
                elif top_issue == 'duration':
                    summary_lines.append("Many notes have duration mismatches — practice sustaining and count subdivision.")
                else:
                    summary_lines.append("Dynamics issues detected — work on consistent touch and clear attacks.")

                # include up to 6 specific examples
                examples = []
                for lst in (pitch_issues, dur_issues, vol_issues, other):
                    for item in lst:
                        if len(examples) >= 6:
                            break
                        if lst is pitch_issues:
                            nm, st, cents = item
                            if cents is None:
                                examples.append(f"{nm} at {st:.2f}s: wrong pitch")
                            else:
                                examples.append(f"{nm} at {st:.2f}s: {cents:+.0f} cents")
                        elif lst is dur_issues:
                            nm, st, diff = item
                            examples.append(f"{nm} at {st:.2f}s: duration off by {diff:.2f}s")
                        elif lst is vol_issues:
                            nm, st, rval = item
                            examples.append(f"{nm} at {st:.2f}s: volume issue")
                        else:
                            nm, st = item
                            examples.append(f"{nm} at {st:.2f}s: issue")
                    if len(examples) >= 6:
                        break

                if examples:
                    summary_lines.append("Examples: " + ", ".join(examples))

                summary = " \n".join(summary_lines)
        except Exception:
            # fallback to the previous simple summary
            if wrong_count == 0:
                summary = "Great job! All notes were played correctly in timing."
            elif ratio > 0.5:
                summary = "Many notes were played incorrectly — try slowing the piece down and isolating the difficult passages."
            elif ratio > 0.2:
                summary = "Some notes were played incorrectly. Focus on the highlighted measures and practice slowly."
            else:
                msgs = [f"{midi_to_note_name(n)} at {st:.2f}s" for n,st in wrong]
                summary = "You missed the following notes: " + ", ".join(msgs) + ". Try slowing down those passages and focus on hand coordination."

        # Generate human-friendly per-note feedback using duration & RMS analysis
        try:
            # Only produce explicit per-note feedback when the note was judged incorrect
            per_note_lines = []
            for ar in analysis_results:
                note, detected_freq, correct, (st,et), dur_est, rms = ar
                # only consider notes that were marked incorrect in the live evaluation
                if correct:
                    continue
                note_name = midi_to_note_name(note)
                expected_dur = et - st
                # volume magnitude: compare rms to a small threshold to decide magnitude
                loud_thresh = 0.04
                vol_line = None
                if rms > loud_thresh * 1.8:
                    vol_line = f"This part could be a lot louder/softer — try playing a lot softer here." if rms > loud_thresh*2 else f"This part could be a little louder/softer."
                elif rms < loud_thresh * 0.6:
                    vol_line = f"This part could be a lot softer/louder — try a lot louder here." if rms < loud_thresh*0.4 else f"This part could be a little louder."
                # duration magnitude
                dur_line = None
                if dur_est > expected_dur * 1.25:
                    mag = 'a lot' if dur_est > expected_dur * 1.5 else 'a little'
                    dur_line = f"This note was played {mag} longer than it should be."
                elif dur_est < expected_dur * 0.7:
                    mag = 'a lot' if dur_est < expected_dur * 0.5 else 'a little'
                    dur_line = f"This note was played {mag} shorter than it should be."
                # pitch check
                pitch_line = None
                if detected_freq:
                    expected_freq = midi_to_freq(note)
                    cents = 1200 * np.log2(detected_freq / expected_freq) if expected_freq>0 else 0
                    if abs(cents) > 30:
                        pitch_line = f"{note_name} was played wrong here (about {cents:+.0f} cents off)."
                    elif abs(cents) > 8:
                        pitch_line = f"{note_name} was slightly out of tune ({cents:+.0f} cents)."
                    else:
                        pitch_line = f"{note_name} was played wrong here."
                else:
                    # if no pitch detected, fall back to a generic accuracy message
                    pitch_line = f"{note_name} was played wrong here."

                # Prefer pitch/accuracy message first, then duration/volume if notable
                chosen_lines = [pitch_line]
                if dur_line:
                    chosen_lines.append(dur_line)
                if vol_line:
                    chosen_lines.append(vol_line)

                # prepend time for clarity
                time_prefixed = [f"{line} ({st:.2f}s)" for line in chosen_lines]
                per_note_lines.extend(time_prefixed)

            if per_note_lines:
                feedback_history.append('--- Detailed Per-note Suggestions ---')
                # keep last N lines to avoid flooding
                feedback_history.extend(per_note_lines[-12:])
                # Optionally refine messages with OpenAI for better phrasing
                try:
                    if openai_available and os.getenv('OPENAI_API_KEY'):
                        prompt = (
                            "Rewrite the following short practice suggestions to be concise, friendly, and actionable:\n" + "\n".join(per_note_lines[-12:])
                        )
                        openai.api_key = os.getenv('OPENAI_API_KEY')
                        resp = openai.ChatCompletion.create(model="gpt-4", messages=[{"role":"user","content":prompt}], max_tokens=300)
                        refined = resp.choices[0].message.content.strip()
                        if refined:
                            # replace the last appended lines with the refined text
                            feedback_history = feedback_history[:-len(per_note_lines[-12:])] + [refined]
                except Exception:
                    pass
        except Exception:
            pass

        # Use OpenAI to generate a final summary if available; provide instrument and detailed per-note issues
        openai_error = None
        try:
            if openai_available and os.getenv('OPENAI_API_KEY'):
                openai.api_key = os.getenv('OPENAI_API_KEY')
                inst = self.instrument.get() if hasattr(self, 'instrument') else 'instrument'
                # Build a compact list of per-note issues from analysis_results
                issue_lines = []
                for ar in analysis_results:
                    note, detected_freq, correct, (st,et), dur_est, rms = ar
                    if correct:
                        continue
                    nm = midi_to_note_name(note)
                    parts = []
                    if detected_freq:
                        expected_freq = midi_to_freq(note)
                        cents = 1200 * np.log2(detected_freq / expected_freq) if expected_freq>0 else 0
                        if abs(cents) > 8:
                            parts.append(f"pitch {cents:+.0f}c")
                    else:
                        parts.append("no pitch")
                    expected_dur = et - st
                    if dur_est > expected_dur * 1.25:
                        parts.append("too long")
                    elif dur_est < expected_dur * 0.7:
                        parts.append("too short")
                    loud_thresh = 0.04
                    if rms > loud_thresh * 1.8:
                        parts.append("too loud")
                    elif rms < loud_thresh * 0.6:
                        parts.append("too quiet")
                    issue_lines.append(f"{nm}@{st:.2f}s: {', '.join(parts)}")

                prompt = (
                    f"You are a helpful practice coach for {inst} players. Below is a list of problematic notes from a recent performance (note@time: issues).\n\n"
                    + "\n".join(issue_lines[:60])
                    + "\n\nProvide 3-6 concise, instrument-specific practice tips focusing on the most common issues listed, and finish with one encouraging sentence tailored to the performer. Keep it friendly and actionable."
                )
                try:
                    resp = openai.ChatCompletion.create(model="gpt-4", messages=[{"role":"user","content":prompt}], max_tokens=400)
                    refined = resp.choices[0].message.content.strip()
                    if refined:
                        summary = refined
                except Exception as e:
                    openai_error = f"OpenAI request failed: {e}"
        except Exception as e:
            openai_error = f"OpenAI initialization failed: {e}"

        # If we had an OpenAI error, show it briefly in the review area for diagnostics
        if openai_error:
            try:
                set_review("AI summary unavailable: " + openai_error)
                # also append to feedback_history so it's visible on Return to Home
                feedback_history.append("(AI summary unavailable: " + openai_error + ")")
                self.music_frame.update()
            except Exception:
                pass

        # Playback finished. User can return home to see the full feedback summary.
        try:
            set_review("Playback complete. Return to Home to view full feedback summary.")
            self.music_frame.update()
        except Exception:
            pass

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
            # clean up playback controls if present
            try:
                if hasattr(self, 'playback_controls'):
                    self.playback_controls.destroy()
            except Exception:
                pass
            try:
                back_btn.destroy()
            except Exception:
                pass
            # show main UI
            self.music_frame.pack_forget()
            self.main_frame.pack(fill='both', expand=True)
            # disable recording nav when returning home so the tab is only available during an active selection/recording flow
            try:
                if hasattr(self, 'nav_record_btn'):
                    try:
                        self.nav_record_btn.state(['disabled'])
                    except Exception:
                        try:
                            self.nav_record_btn.config(state='disabled')
                        except Exception:
                            pass
            except Exception:
                pass
            # update the Feedback section on the main screen (multi-line)
            accuracy = f"Final Accuracy: {100*sum(1 for ok in evaluation_results.values() if ok)/max(1,len(evaluation_results)):.1f}%"
            fb_lines = []
            fb_lines.append("Recording complete!")
            if getattr(self, 'downloaded_last', False):
                fb_lines.append(f"Saved to: {wav_path}")
            fb_lines.append(accuracy)
            # include the AI summary if available
            try:
                fb_lines.append('')
                fb_lines.append('Summary:')
                fb_lines.append(summary)
            except Exception:
                pass
            try:
                # Compose a readable multi-line message for the feedback label
                label_lines = fb_lines
                try:
                    # include latest per-note lines if available
                    label_lines.append("")
                    label_lines.append('Recent notes:')
                    label_lines.extend(feedback_history[-12:])
                except Exception:
                    pass
                self.feedback_label.config(text="\n".join(label_lines))
            except Exception:
                pass

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
