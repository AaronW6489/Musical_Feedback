import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import librosa
import sounddevice as sd
from utils import midi_to_note_name


def open_tuner_window(parent):
	win = tk.Toplevel(parent)
	win.title("Tuner")
	win.geometry("420x320")
	ttk.Label(win, text="Tuner (select instrument and string)").pack(pady=8)
	inst_var = tk.StringVar(value="cello")
	ttk.Radiobutton(win, text="Cello", variable=inst_var, value='cello').pack(anchor='w')
	ttk.Radiobutton(win, text="Guitar", variable=inst_var, value='guitar').pack(anchor='w')

	string_frame = ttk.Frame(win)
	string_frame.pack(pady=8)

	cello_strings = {'C':36, 'G':43, 'D':50, 'A':57}
	guitar_strings = {'E':40, 'A':45, 'D':50, 'G':55, 'B':59, 'E2':64}

	target_note_var = tk.IntVar(value=60)
	status_var = tk.StringVar(value='Idle')
	ttk.Label(win, textvariable=status_var, font=('Segoe UI', 12, 'bold')).pack(pady=6)

	def select_string(note_midi):
		target_note_var.set(note_midi)

	def rebuild_string_buttons():
		for w in string_frame.winfo_children():
			w.destroy()
		if inst_var.get() == 'cello':
			for name, midi in cello_strings.items():
				ttk.Button(string_frame, text=name, command=lambda m=midi: select_string(m)).pack(side='left', padx=4)
		else:
			for name, midi in guitar_strings.items():
				ttk.Button(string_frame, text=name, command=lambda m=midi: select_string(m)).pack(side='left', padx=4)

	inst_var.trace_add('write', lambda *a: rebuild_string_buttons())
	rebuild_string_buttons()

	pitch_label = ttk.Label(win, text='Freq: -- Hz | Note: -- | Cents: --')
	pitch_label.pack(pady=8)

	meter_canvas = tk.Canvas(win, width=360, height=50, bg='black', highlightthickness=0)
	meter_canvas.pack(pady=6)
	meter_center = 180
	meter_canvas.create_line(meter_center, 0, meter_center, 50, fill='white')
	meter_indicator = meter_canvas.create_rectangle(meter_center-3, 10, meter_center+3, 40, fill='lime')

	stop_event = threading.Event()

	def tuner_loop():
		fs = 44100
		try:
			with sd.InputStream(samplerate=fs, channels=1) as stream:
				recent_freqs = []
				while not stop_event.is_set():
					try:
						data, _ = stream.read(2048)
					except Exception:
						continue
					if data is None or len(data) == 0:
						continue
					y = data[:,0].astype(float)
					L = len(y)
					if L < 32:
						freq = 0.0
					else:
						n_fft = 2 ** int(np.floor(np.log2(L)))
						n_fft = min(n_fft, 2048)
						try:
							import warnings
							with warnings.catch_warnings():
								warnings.simplefilter('ignore')
								pitches, mags = librosa.piptrack(y=y, sr=fs, n_fft=n_fft, hop_length=max(64, n_fft//4), fmin=50, fmax=2000)
							if mags.size == 0:
								freq = 0.0
							else:
								idx = mags.argmax()
								i, j = np.unravel_index(idx, mags.shape)
								freq = float(pitches[i, j])
						except Exception:
							freq = 0.0

					if freq and freq > 0:
						recent_freqs.append(freq)
						if len(recent_freqs) > 8:
							recent_freqs.pop(0)
						med = float(np.median(recent_freqs))
					else:
						med = 0.0

					if med and med > 0:
						midi = int(round(69 + 12 * np.log2(med / 440.0)))
						target = target_note_var.get()
						try:
							cents = 1200 * np.log2(med / (440.0 * 2**((target-69)/12))) if target else 0.0
						except Exception:
							cents = 0.0
						txt = f'Freq: {med:.1f} Hz | Note: {midi_to_note_name(midi)} | Cents: {cents:+.1f}'
					else:
						txt = 'Freq: -- Hz | Note: -- | Cents: --'

					def safe_update(t=txt, c=cents if 'cents' in locals() else None):
						try:
							pitch_label.config(text=t)
							if c is not None:
								cc = max(-100, min(100, c))
								pos = meter_center + (cc / 100.0) * (meter_center - 20)
								meter_canvas.coords(meter_indicator, pos-4, 10, pos+4, 40)
						except Exception:
							pass

					pitch_label.after(0, safe_update)
		except Exception:
			def safe_err():
				try:
					pitch_label.config(text='Audio input unavailable')
				except Exception:
					pass
			pitch_label.after(0, safe_err)

	thread = threading.Thread(target=tuner_loop, daemon=True)
	thread.start()

	def stop_tuner():
		stop_event.set()
		win.destroy()

	ttk.Button(win, text='Stop Tuner', command=stop_tuner).pack(pady=8)

