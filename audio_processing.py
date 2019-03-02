import os
import scipy

from pydub import AudioSegment
from scipy.io import wavfile
import tempfile
import os
import scipy
import scipy.io.wavfile


def get_audio_array_audio_segment(path):
    return AudioSegment.from_mp3(path)


def get_data(filename):
    fs, data = wavfile.read(filename)
    print(fs, data, sep='\n')


def get_audio_array_process(path):  # works but is not tested
    import subprocess as sp

    FFMPEG_BIN = "ffmpeg"

    command = [FFMPEG_BIN,
               '-i', path,
               '-f', 's16le',
               '-acodec', 'pcm_s16le',
               '-ar', '44100',  # ouput will have 44100 Hz
               '-ac', '2',  # stereo (set to '1' for mono)
               '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)
    raw_audio = pipe.stdout.read(88200 * 4)

    import numpy

    audio_array = numpy.fromstring(raw_audio, dtype="int16")
    if len(audio_array) % 2 == 1:
        del audio_array[-1]
    audio_array = audio_array.reshape((int(len(audio_array) / 2), 2))
    return audio_array


