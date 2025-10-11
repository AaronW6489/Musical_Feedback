import threading
import numpy as np
import sounddevice as sd
import tempfile
import time


def start_metronome(stop_event, bpm, fs=44100, start_time=None, duration=None):
	"""Start a daemon thread that plays a short metronome 'tap' at `bpm` until `stop_event` is set.
	If start_time is provided (epoch seconds), scheduling aligns to that start using perf_counter.
	Returns the Thread object.
	"""
	def _loop(duration=None):
		interval = 60.0 / float(max(1.0, bpm))
		# very short click (sine burst) with quick exponential envelope
		duration_tap = 0.02
		t = np.linspace(0, duration_tap, int(fs*duration_tap), endpoint=False)
		env = np.exp(-80 * t)
		click = (0.8 * np.sin(2*np.pi*2000*t) * env).astype('float32')

		logfile = tempfile.gettempdir() + "/metronome_log.txt"

		# If duration provided, precompute full buffer for reliability and play once
		if duration is not None and duration > 0:
			try:
				total_samples = int(max(1, duration * fs))
				buf = np.zeros(total_samples, dtype='float32')
				# schedule clicks at intervals aligned to start_time (if provided) or immediate
				if start_time is not None:
					# compute offset seconds from now until start_time
					offset = max(0.0, start_time - time.time())
				else:
					offset = 0.0
				first_idx = int(round(offset * fs))
				idx = first_idx
				step = int(round(interval * fs))
				while idx < total_samples:
					end = min(total_samples, idx + len(click))
					buf[idx:end] += click[:end-idx]
					idx += step
				try:
					sd.play(buf, fs)
				except Exception:
					# fallback: play click repeatedly
					idx = first_idx
					while idx < total_samples and not stop_event.is_set():
						try:
							sd.play(click, fs)
							sd.wait()
						except Exception:
							pass
						time.sleep(interval)
				# wait until buffer finishes or stop_event
				start = time.time()
				while time.time() - start < duration and not stop_event.is_set():
					time.sleep(0.05)
				try:
					sd.stop()
				except Exception:
					pass
				return
			except Exception:
				# fall through to streaming loop
				pass

		# streaming fallback: write clicks on schedule using OutputStream
		try:
			with sd.OutputStream(samplerate=fs, channels=1, dtype='float32') as out:
				next_time = time.perf_counter()
				if start_time is not None:
					# align perf_counter to epoch start_time
					next_time = time.perf_counter() + max(0.0, start_time - time.time())
				while not stop_event.is_set():
					now = time.perf_counter()
					if now >= next_time:
						try:
							out.write(click.reshape(-1,1))
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
			# last-resort fallback: sd.play single click and sleep
			while not stop_event.is_set():
				try:
					sd.play(click, fs)
					sd.wait()
				except Exception:
					pass
				time.sleep(interval)

	thread = threading.Thread(target=_loop, args=(duration,), daemon=True)
	thread.start()
	return thread

