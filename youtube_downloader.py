from pytube import YouTube
from pytube import Playlist
from time import time
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


def download_video(url, path):
    yt = YouTube(url)
    streams = yt.streams.filter(res='1080p')
    video = streams.first()
    video.download(path)


def download_to_mp3(url, path):
    yt = YouTube(url)
    streams = yt.streams.filter(only_audio=True)
    video = streams.first()
    video.download(path)


def download_playlist(url, path):
    pl = Playlist(url)
    pl.download_all(path)


start_time = time()
costel_path = 'videos'
url = 'https://www.youtube.com/watch?v=p4dvTQB_JHA'
download_to_mp3(url, costel_path)
print('Finished download in {} minutes'.format((time()-start_time)/60))
