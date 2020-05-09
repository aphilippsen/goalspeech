import numpy as np
import struct
import wave
import sounddevice

def write_wav_file(filename, audio, fs):
    outputFile = wave.open(filename, 'wb')
    outputFile.setparams((1, 2, fs, 0, 'NONE', 'not compressed'))

    if np.abs(np.max(audio)) <= 1:
        audio = [int(32767.0 * a) for a in audio]

    if -32767 > np.min(audio) or np.max(audio) > 32767:
        print("Warning in write_wav_file(): Value in audio signal out of range! Audio can not be packed into short format, max value " + str(np.max(audio)) + ", min value " + str(np.min(audio)) + '\n')

    for i in range(0, np.size(audio)):
        value = audio[i] #int(32767.0 * audio[i])
        # overflow happened
        if -32767 > value or value > 32767:
            value = 0
        #print(value)
        packed_value = struct.pack('<h', value)

        outputFile.writeframes(packed_value)

    outputFile.close()

def play_wav_file(audio, fs=22050, blocking=True):
    sounddevice.default.samplerate = fs
    sounddevice.play(audio, blocking=blocking)
