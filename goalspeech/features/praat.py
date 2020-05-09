import logging
import os
import numpy as np
from subprocess import Popen, PIPE
import goalspeech

from goalspeech.utils.wav import write_wav_file

praat_script_location = os.path.join(goalspeech.__path__[0], "features/praatScripts/")

def getPraatFormants(sound, fs, speakerType = 'male', tmpDir = "."):
    """
        Calls praat to calculate formants for sound (np array). According
        to speaker type, maxFreq is adapted to a meaningful value.
        In tmpDir, the temporary wav file is produced for passing it to praat.
    """

    try:
        praatCmd = os.environ['PRAAT_CMD']
    except:
        raise ValueError("Environment variable PRAAT_CMD missing!")

    if not praatCmd:
        praatCmd = 'praat'

    try:
        if not os.path.exists(tmpDir):
            os.mkdir(tmpDir)
    except:
        print("Could not create tmp dir: " + tmpDir)

    praatFormantsScript = os.path.join(praat_script_location, 'formants.praat')

    tmpWavFile = os.path.abspath(os.path.join(tmpDir, "current.wav"))
    write_wav_file(tmpWavFile, sound, fs)

    # do subprocess communication
    numFormantsCalc = 5
    numFormants = 3

    if speakerType == 'male':
        ceilingFormant = 4000
    elif speakerType == 'child':
        ceilingFormant = 8000
    else:
        print("Undefined speakerType: " + speakerType)

    cmd = [praatCmd, '--run', praatFormantsScript, tmpWavFile, str(numFormantsCalc), str(ceilingFormant)]
    #print(cmd)
    p = Popen(cmd,
              shell=False, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()

    try:
        exitcode = p.wait()
    except KeyboardInterrupt:
        exitcode = p.wait()
    if exitcode != 0:
        errmsg = "%s exited with non-zero status" % praatCmd
        logging.error(errmsg)

    timeFormantArr = np.array([tryFloat(x) for x in stdout.split()[numFormants+1:len(stdout.split())]])
    timeFormantArr = timeFormantArr.reshape(-1, numFormants+1)

    timeStamps = timeFormantArr[:,0]
    formants = timeFormantArr[:,1:numFormants+1]

    # set value from earlier frame, if --undefined-- occurs
    for i in range(np.size(formants,0)):
        try:
            formants[i,formants[i,:] == 0] = formants[i-1,formants[i,:] == 0]
        except:
            pass

    return (timeStamps, formants)

def tryFloat(x):
    try:
        y = float(x)
    except:
        y = 0
    return y


def getPraatFormantsMean(sound, fs, speakerType = 'male', tmpDir = "."):
    """
        Calls praat to calculate formants for sound (np array). According
        to speaker type, maxFreq is adapted to a meaningful value.
        In tmpDir, the temporary wav file is produced for passing it to praat.
    """

    all_formants = getPraatFormants(sound, fs, speakerType, tmpDir)
    return np.mean(all_formants[1], axis=0)
