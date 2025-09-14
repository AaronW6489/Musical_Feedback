import threading
import numpy as np
import sounddevice as sd
import tempfile
import time


def start_metronome(stop_event, bpm, fs=44100):
	"""Start a daemon thread that plays a metronome beep at `bpm` until `stop_event` is set.
	Also logs scheduled vs actual tick times to the system temp folder for diagnostics.
	Returns the Thread object.
	"""
	def _loop():
		interval = 60.0 / float(max(1.0, bpm))
		duration_beep = 0.06
		t = np.linspace(0, duration_beep, int(fs*duration_beep), endpoint=False)
		env = np.exp(-30 * t)
		beep = (0.6 * np.sin(2*np.pi*200*t) * env).astype('float32')
		logfile = tempfile.gettempdir() + "/metronome_log.txt"
		try:
			with sd.OutputStream(samplerate=fs, channels=1, dtype='float32') as out:
				next_time = time.perf_counter()
				while not stop_event.is_set():
					now = time.perf_counter()
					if now + 0.001 >= next_time:
						try:
							out.write(beep.reshape(-1,1))
						except Exception:
							pass
						try:
							with open(logfile, 'a') as lf:
								lf.write(f"{time.time():.6f},{next_time:.6f},{now:.6f},{(now-next_time):.6f}\n")
								lf.flush()
						except Exception:
							pass
						next_time += interval
					else:
						time.sleep(max(0.0005, next_time - now))
		except Exception:
			while not stop_event.is_set():
				try:
					sd.play(beep, fs)
					sd.wait()
				except Exception:
					pass
				time.sleep(interval)

	thread = threading.Thread(target=_loop, daemon=True)
	thread.start()
	return thread

