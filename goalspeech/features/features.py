import numpy as np

from goalspeech.features.praat import getPraatFormants, getPraatFormantsMean
from python_speech_features.base import mfcc, logfbank

def calcAcousticFeatures(sound, fs, featureMode, speakerType = 'male', tmpDir = ".", speech_sound_type='vowel', octave_binding = None):
    """
        Calculates acoustic features with given featureMode for sound
        with audio sampling rate fs.

        sound: 1D np.array or 2D array of horizontal vector.

        Returns a tuple containing a 2D np.array of numTimeSteps x numFeatureParams
        and a 2 x numFeatureParams array that contains scaling factors that
        can be used to ensure equal contribution of each feature type.
    """

    # how many Hz may change in the first formant in mergeFactor ms – few variation for vowels required
    if speech_sound_type == 'vowel':
        maxFormantChange = 50
    elif speech_sound_type == 'syllable':
        maxFormantChange = 800

    if np.ndim(sound) == 2:
        sound = sound[0,:]

    if featureMode == 'formants':
        formants = getPraatFormantsMean(sound, fs, speakerType, tmpDir)
        return (np.array(formants).reshape((1,-1)), None)

    elif featureMode == 'formants_full':
        (timePos, formants) = getPraatFormants(sound, fs, speakerType, tmpDir)

        # downsample
        mergeFactor=10 # how many time steps (=ms) should be merged to one value
        newFormants = np.array(np.mean(formants[0:mergeFactor,:], 0), ndmin=2)
        for t in range(mergeFactor, len(timePos), mergeFactor):
            new = np.mean(formants[t:t+mergeFactor,:], 0)

            if abs(newFormants[-1, 0] - new[0]) > maxFormantChange:
                # TODO: this is dangerous if the first detected formant is incorrect!
                pass
            else:
                newFormants = np.vstack((newFormants, new))

        return (newFormants, None)

    elif featureMode == 'mfcc':
        # sound as 1d
        if sound.ndim == 2:
            sound = np.reshape(sound, (-1))

        # returns (numFrames x numCeps) np array
        window_length = 0.02 # 0.025 * 22050 = ca. 551 frames
        window_step = 0.01 # 0.01 * 22050 = ca. 221 frames
        num_cepstrals = 13
        features = mfcc(sound, fs, window_length, window_step, num_cepstrals)
        return (features, None)

    elif featureMode == 'mfcc_formants':

        # sound as 1d
        if sound.ndim == 2:
            sound = np.reshape(sound, (-1))

        # returns (numFrames x numCeps) np array
        window_length = 0.02 # 0.025 * 22050 = ca. 551 frames
        window_step = 0.005 # 0.01 * 22050 = ca. 221 frames
        num_cepstrals = 13
        features = mfcc(sound, fs, window_length, window_step, num_cepstrals)

        (timePos, formants) = getPraatFormants(sound, fs, speakerType, tmpDir)
        # downsample
        mergeFactor=10 # how many time steps (=ms) should be merged to one value

        # get a good estimate for initial formants (ignoring initial perturbations):
        initialFormants = np.median(formants[0:5,:],axis=0)
        newFormants = None
        i = 0
        while not newFormants:
            if abs(formants[i,0] - initialFormants[0]) < maxFormantChange:
                newFormants = formants[i,:]
                break
            else:
                i += 1

        newFormants = np.array(np.mean(formants[0:mergeFactor,:], 0), ndmin=2)
        for t in range(mergeFactor, len(timePos), mergeFactor):
            new = np.mean(formants[t:t+mergeFactor,:], 0)
            if abs(newFormants[-1, 0] - new[0]) > maxFormantChange:
                pass
            else:
                newFormants = np.vstack((newFormants, new))

        # resample formants according to mfccs
        # TODO Warning: interp just copies the last element to make trajectories longer!!!
        # alternative: https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
        resampledFormants = np.zeros((np.shape(features)[0], np.shape(newFormants)[1]))
        for i in range(np.shape(newFormants)[1]):
            resampledFormants[:,i] = np.interp(range(np.shape(features)[0]), range(np.shape(newFormants)[0]), newFormants[:,i])

        minmax = np.array([np.concatenate((np.repeat([-1/np.shape(resampledFormants)[1]], np.shape(resampledFormants)[1]), np.repeat([-1/np.shape(features)[1]], np.shape(features)[1]))), np.concatenate((np.repeat([1/np.shape(resampledFormants)[1]], np.shape(resampledFormants)[1]), np.repeat([1/np.shape(features)[1]], np.shape(features)[1])))])
        return (np.concatenate((resampledFormants, features), axis=1), minmax)

    elif featureMode == "fbank":

        # sound as 1d
        if sound.ndim == 2:
            sound = np.reshape(sound, (-1))
        fbank_feat = logfbank(sound, fs, nfft=1024)

        return (fbank_feat, np.concatenate(([-1*np.ones(fbank_feat.shape[1])], [np.ones(fbank_feat.shape[1])])))

    elif featureMode == "logfbank":

        # sound as 1d
        if sound.ndim == 2:
            sound = np.reshape(sound, (-1))
        fbank_feat = logfbank(sound, fs, nfft=1024)

        return (fbank_feat, np.concatenate(([-1*np.ones(fbank_feat.shape[1])], [np.ones(fbank_feat.shape[1])])))

    elif featureMode == 'gbfb': # Gabor filter bank features, requires Octave installed

        # scaledAudio = np.int16(copiedAudio/maxAmplitude * 32767)
        soundNorm = sound/32767

        #features = octave_binding.gbfb_feature_extraction(soundNorm, fs)
        features = octave_binding.heq(octave_binding.gbfb(octave_binding.log_mel_spectrogram(soundNorm, fs)))
        
        features = features.transpose()
        return (features, np.concatenate(([-1*np.ones(features.shape[1])], [np.ones(features.shape[1])])))


    else:
        print("Feature mode " + featureMode + " not yet defined in calcAcousticFeatures()!")

    return None
