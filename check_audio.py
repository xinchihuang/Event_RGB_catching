import os

path = '/dev/snd/controlC0'
try:
    fd = os.open(path, os.O_RDWR)
    print('open ok', fd)
    os.close(fd)
except OSError as e:
    print('open failed', e.errno, e)
