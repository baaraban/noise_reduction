import wave
import numpy as np


def get_wav_info(name):
    with wave.open(name, 'r') as wf:
        return wf.readframes(-1), wf.getnchannels(), wf.getframerate()


def get_audio_matrix(frames, frame_rate , ms_per_column = 500):
    def append_zeros(arr, n_zeros):
        return np.pad(arr, (0, n_zeros), mode='constant', constant_values=(0))

    frames_per_column = int(frame_rate * (ms_per_column / 1000))
    columns = int(np.ceil(len(frames) / frames_per_column))
    to_append = frames_per_column * columns - len(frames)

    return append_zeros(frames, to_append).reshape((frames_per_column, columns))


def write_wav_file(filename, frames, n_channels, frame_rate, compression_type = "NONE", compression_name= "Uncompressed"):
    with wave.open(filename, 'w') as f:
        f.setparams((n_channels, frames.dtype.itemsize, frame_rate, len(frames), compression_type, compression_name))
        f.writeframes(frames.tostring())