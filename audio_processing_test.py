import os
import scipy
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
import tempfile
import os
import scipy
import scipy.io.wavfile

from audio_processing import get_audio_array_audio_segment, get_audio_array_process
from utils import get_data, get_sound_array_interval, save_obj


def test1():
    costel_path = 'videos/aa.mp3'
    sound = get_audio_array_audio_segment(costel_path)

    halfway_point = len(sound) / 2
    second_half = sound[halfway_point:]
    second_half_3_times = second_half + second_half + second_half

    second_half_3_times.export('videos/Costel2.mp3', format='mp3')


def test2():
    sound = get_audio_array_process('videos/aa.mp3')
    halfway_point = len(sound) // 2
    second_half = sound[halfway_point:]
    second_half_3_times = second_half + second_half + second_half

    second_half_3_times.export('videos/bb.mp3', format='mp3')



sound = get_audio_array_process('videos/aa.mp3')
entries = []
data = get_data()
for entry in data:
     label = np.zeros(3)
     start, end, l = entry
     label[l] = 1
     for i in range(start, end):
        instance = get_sound_array_interval(sound, i, i+1)
        entries.append((instance, label))
entries = np.array(entries)
save_obj(entries, 'entries.pickle')

