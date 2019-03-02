import pickle
import numpy as np

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_data():
    path = 'log.txt'
    names = 'tvc'
    data = []
    with open(path, 'r') as handle:
        for line in handle.readlines():
            l = line.split(' ')
            if len(l) == 3:
                start = l[0]
                end = l[1]
                p = l[2]
                if len(start) < 4:
                    start = '0' + start
                if len(end) < 4:
                    end = '0' + end
                secs = int(start[2:])
                mins = int(start[:2])
                start = mins * 60 + secs

                secs = int(end[2:])
                mins = int(end[:2])
                end = mins * 60 + secs
                p = p.strip()
                data.append((start, end, names.index(p)))
    data = np.array(data)
    return data


mp3_time_length_secs = 34 * 60 + 45


def get_sound_array_interval(p_sound, start_sec, end_sec):
    start_i = int((start_sec/mp3_time_length_secs) * len(p_sound))
    end_i = int((end_sec/mp3_time_length_secs) * len(p_sound))
    return p_sound[start_i:end_i]