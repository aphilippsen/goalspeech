import ctypes            # for accessing C libraries
import itertools         # for iterating over nested lists
import numpy as np       # for array operations
import os                # for retrieving path names
import scipy.io.wavfile  # for writing wav file
import sounddevice as sd # for playing back audio
from tqdm import tqdm    # progress bar for sound production

class VTLSpeechProduction:

    def __init__(self, library, speaker):
        self.speaker = speaker
        self.speakerFileName = os.path.abspath("libs/VTL2.1_Linux/" + speaker + ".speaker")

        self.library = library
        if self.library == 'vtl-2.1':
            dllFile = os.path.abspath("libs/VTL2.1_Linux/VocalTractLabApi.so")
            self.lib = ctypes.cdll.LoadLibrary(dllFile)
        elif self.library == 'vtl-2.1b':
            dllFile = os.path.abspath("libs/vtlapi-2.1b/VocalTractLabApi64.so")
            self.lib = ctypes.cdll.LoadLibrary(dllFile)

        # init VTL library
        self.lib.vtlInitialize(self.speakerFileName.encode('ascii'))

        self.arSamplingRate = 200

        # get vtl constants
        c_int_ptr = ctypes.c_int * 1 # type for int*
        audioSamplingRate_ptr = c_int_ptr(0)
        numTubeSections_ptr = c_int_ptr(0)
        numVocalTractParams_ptr = c_int_ptr(0)
        numGlottisParams_ptr = c_int_ptr(0)
        self.lib.vtlGetConstants(audioSamplingRate_ptr, numTubeSections_ptr, numVocalTractParams_ptr, numGlottisParams_ptr)
        self.audioSamplingRate = audioSamplingRate_ptr[0]
        self.numTubeSections = numTubeSections_ptr[0]
        self.numVocalTractParams = numVocalTractParams_ptr[0]
        self.numGlottisParams = numGlottisParams_ptr[0]

        # get tract info
        c_numTractParam_ptr = ctypes.c_double * self.numVocalTractParams
        tractParamNames = ctypes.create_string_buffer(self.numVocalTractParams * 32)
        self.tractParamMin = c_numTractParam_ptr(0)
        self.tractParamMax = c_numTractParam_ptr(0)
        self.tractParamNeutral = c_numTractParam_ptr(0)
        self.lib.vtlGetTractParamInfo(tractParamNames, self.tractParamMin, self.tractParamMax, self.tractParamNeutral)
        self.tractParamMin = np.array(self.tractParamMin)
        self.tractParamMax = np.array(self.tractParamMax)
        self.tractParamNeutral = np.array(self.tractParamNeutral)

        # get glottis info
        c_numGlottisParam_ptr = ctypes.c_double * self.numGlottisParams
        glottisParamNames = ctypes.create_string_buffer(self.numGlottisParams * 32)
        self.glottisParamMin = c_numGlottisParam_ptr(0)
        self.glottisParamMax = c_numGlottisParam_ptr(0)
        self.glottisParamNeutral = c_numGlottisParam_ptr(0)
        self.lib.vtlGetGlottisParamInfo(glottisParamNames, self.glottisParamMin, self.glottisParamMax, self.glottisParamNeutral)
        self.glottisParamMin = np.array(self.glottisParamMin)
        self.glottisParamMax = np.array(self.glottisParamMax)
        self.glottisParamNeutral = np.array(self.glottisParamNeutral)

        # adapt neutral params of glottis for better sound in version 2.1
        if self.library == 'vtl-2.1':
            if self.speaker == 'JD2':
                # triangular glottis
                self.glottisParamNeutral = np.array([103.826, 700, 0.000147, 9.5e-05, 0, -40])
            else:
                # glottis titze
                self.glottisParamNeutral = np.array([120, 800, 0.00030, 0.00030, 0.000000, 0.880000,-35.000000])
        elif self.library == 'vtl-2.1b':
            self.glottisParamNeutral = np.array([103.826, 700, 0.000147, 9.5e-05, 0, -40])

        sd.default.samplerate = self.audioSamplingRate

    def getArticulatoryShape(self, shapeName):
        """ """
        c_numTractParam_ptr = ctypes.c_double * self.numVocalTractParams # double* with enough space for all VT params

        # get shape information from speaker
        vtParams = c_numTractParam_ptr(0)
        failure = self.lib.vtlGetTractParams(shapeName.encode('ascii'), vtParams)

        if (failure != 0):
            print("Error: Vocal tract shape '" + shapeName + "' not in the speaker file: " + self.speakerFileName)

        return np.array(np.hstack((vtParams, self.glottisParamNeutral)), dtype=float)

    def getArticulatoryShapes(self, sequences):
        shapes = np.empty(np.size(sequences), dtype=object)
        for (i,s) in zip(range(len(sequences)), sequences):
            shapes[i] = self.getArticulatoryShape(s)
        return shapes

    def getArticulatoryTrajectory(self, sequence, gestureFileDir):
        """ """
        gestureFile = os.path.abspath(os.path.join(gestureFileDir, sequence + ".ges"))
        wavFileName = os.path.abspath(os.path.join(gestureFileDir, sequence + ".wav"))
        areaFileName = os.path.abspath(os.path.join(gestureFileDir, sequence + ".txt"))

        if not os.path.exists(areaFileName):
            print("Calling vtlGesToWav() for sequence " + sequence)
            failure = self.lib.vtlGesToWav(self.speakerFileName.encode('ascii'), gestureFile.encode('ascii'), wavFileName.encode('ascii'), areaFileName.encode('ascii'))
            if (failure != 0):
                print("Error in vtlGesToWav()!")
            self.repairWavHeader(wavFileName)

        f = open(areaFileName)
        while not f.readline() == "#data\n":
            pass

        all_parameters = []

        while f.readline(): # reads the next timestep
            areas = f.readline()
            lengths = f.readline()
            tract_params = f.readline()
            glottis_params = f.readline()
            all_parameters.append(tract_params.split(' ')[:-1] + glottis_params.split(' ')[:-1])

        parameter_array = np.array(all_parameters,dtype='float')
        # this parameter array has one entry every 10ms, so the sampling rate is 100
        # if the desired ar sampling rate is different, resample here!
        if self.arSamplingRate != 100:
            num_samples = parameter_array.shape[0]
            duration = num_samples / 100
            num_samples_new = int(np.floor(duration * self.arSamplingRate))
            parameter_array_new = np.zeros((num_samples_new, parameter_array.shape[1]))
            for i in range(parameter_array.shape[1]):
                parameter_array_new[:,i] = np.interp(np.arange(0, num_samples, 100/self.arSamplingRate), np.arange(num_samples), parameter_array[:,i])

        return parameter_array_new

    def getArticulatoryTrajectories(self, sequences, gestureFileDir):
        trajectories = np.empty(np.size(sequences), dtype=object)
        for (i,s) in zip(range(len(sequences)), sequences):
            trajectories[i] = self.getArticulatoryTrajectory(s, gestureFileDir)
        return trajectories

    def getExampleArTraj(self):
        """ An example trajectory for testing with the child-1y speaker. """
        c_int_ptr = ctypes.c_int * 1 # type for int*
        c_numTractParam_ptr = ctypes.c_double * self.numVocalTractParams;

        # get shape information from speaker
        shapeName = 'a';
        paramsA = c_numTractParam_ptr(0);
        failure = self.lib.vtlGetTractParams(shapeName.encode('ascii'), paramsA);

        if (failure != 0):
            print("Error: Vocal tract shape 'a' not in the speaker file!")

        shapeName = 'i';
        paramsI = c_numTractParam_ptr(0);
        failure = self.lib.vtlGetTractParams(shapeName.encode('ascii'), paramsI);

        if (failure != 0):
            print("Error: Vocal tract shape 'i' not in the speaker file!")

        duration_s = 1.0;
        numFrames = round(duration_s * self.arSamplingRate);

        tractParams = np.zeros(shape = (numFrames, self.numVocalTractParams), dtype = np.float)
        glottisParams = np.zeros(shape = (numFrames, self.numGlottisParams), dtype = np.float)

        # Create the vocal tract shapes that slowly change from /a/ to /i/ from the
        # first to the last frame.
        for i in range(0, int(numFrames)):
            # The VT shape changes from /a/ to /i/.
            d = i / (numFrames-1)
            for j in range(0, len(paramsA)):
                tractParams[i, j] = (1-d) * paramsA[j] + d * paramsI[j]


            # Set F0 in Hz.
            for j in range(0, self.numGlottisParams):
                # Take the neutral settings for the glottis here.
                glottisParams[i, j] = self.glottisParamNeutral[j]

            glottisParams[i, 0] = 300.0 - 20*(i/numFrames); # ONLY FOR CHILD VT
            glottisParams[i, 1] = 1000.0;

        return np.hstack((tractParams, glottisParams))

    def produceSound(self, params, play_sound = False):
        """ produces sound from tractParameters in VTL
            arTraj is of size (timeSteps x (numTractParams + numGlottisParams))
            returns the scaled audio as produced by VTL """

        if params.dtype == object:
            audio = np.empty(np.size(params), dtype = object)
            for i in tqdm(range(np.size(params))):
                audio[i] = self.produceSound(params[i], play_sound)
            return audio

        tractParams = params[:,0:self.numVocalTractParams]
        glottisParams = params[:,self.numVocalTractParams:self.numVocalTractParams+self.numGlottisParams]

        # transform to python lists 1 x (numFrames * numParams)
        tr = list(itertools.chain.from_iterable(tractParams.tolist()))
        gl = list(itertools.chain.from_iterable(glottisParams.tolist()))
        numFrames = tractParams.shape[0]

        wavFileName = os.path.abspath("result.wav")

        c_tractSequence_ptr = ctypes.c_double * int(numFrames * self.numVocalTractParams)
        c_glottisSequence_ptr = ctypes.c_double * int(numFrames * self.numGlottisParams)
        tube_articulators_ptr = ctypes.c_char_p(b' ' * numFrames * self.numTubeSections)

        # Init arrays
        c_int_ptr = ctypes.c_int * 1 # type for int*
        tractParams_ptr = c_tractSequence_ptr(*tr)
        glottisParams_ptr = c_glottisSequence_ptr(*gl)
        c_tubeAreas_ptr = ctypes.c_double * int(numFrames * self.numTubeSections);
        tubeAreas = c_tubeAreas_ptr(0);
        # 2000 samples more in the audio signal for safety.
        duration_s = numFrames / self.arSamplingRate
        c_audio_ptr = ctypes.c_double * int(duration_s * self.audioSamplingRate + 2000)
        audio = c_audio_ptr(0);
        numAudioSamples = c_int_ptr(0);

        if self.library == 'vtl-2.1':
            failure = self.lib.vtlSynthBlock(ctypes.byref(tractParams_ptr),  # input
                                             ctypes.byref(glottisParams_ptr),  # input
                                             ctypes.byref(tubeAreas),  # output
                                             numFrames,  # input
                                             ctypes.c_double(self.arSamplingRate),  # input
                                             ctypes.byref(audio),  # output
                                             ctypes.byref(numAudioSamples))  # output

        elif self.library == 'vtl-2.1b':
            failure = self.lib.vtlSynthBlock(ctypes.byref(tractParams_ptr),  # input
                                             ctypes.byref(glottisParams_ptr),  # input
                                             ctypes.byref(tubeAreas),  # output
                                             tube_articulators_ptr,  # output
                                             numFrames,  # input
                                             ctypes.c_double(self.arSamplingRate),  # input
                                             ctypes.byref(audio),  # output
                                             ctypes.byref(numAudioSamples))  # output

        if failure != 0:
            raise ValueError("Error in vtlSynthBlock! Errorcode: %i" % failure)

        copiedAudio = np.zeros(shape=(len(audio),1), dtype=np.float)
        for i in range(0, len(audio)):
            copiedAudio[i] = audio[i]

        # VTL sometimes produces strange onset noise, therefore, throw away the first 50 frames
        copiedAudio = copiedAudio[50:len(copiedAudio)]

        # normalize audio and scale to int16 range
        assert(np.isfinite(copiedAudio).all())
        maxAmplitude = np.max(np.abs(copiedAudio))
        if maxAmplitude == 0:
            print("Warning in VTLSpeechProduction: Audio seems to be empty! Do not scale audio.")
            scaledAudio = copiedAudio
            #raise Exception()
        else:
            scaledAudio = np.int16(copiedAudio/maxAmplitude * 32767)
        # write wave file
        # scipy.io.wavfile.write(wavFileName, self.audioSamplingRate, scaledAudio)
        # play wave file
        if play_sound:
            sd.play(scaledAudio, blocking=True)
        return scaledAudio[:,0] # one-dimensional array

    def repairWavHeader(self, filename):
        """
            vtlGesToWav function generates erroneous wav files which this
            function can repair.
        """
        if not os.path.exists(filename):
            print("Error: " + filename + " not found!")
        else:
            # create backup of wav file
            backup = filename+'.bkup'
            os.system('cp ' + filename + ' ' + backup)

            # read wav file content
            f = open(filename, 'rb')
            content = f.read()
            f.close()
            # remove wav file and create empty one
            os.remove(filename)
            os.system('touch '+filename)

            # write corrected header + sound data
            try:
                newfile = open(filename, 'wb')
                header = b'RIFF\x8c\x87\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"V\x00\x00D\xac\x00\x00\x02\x00\x10\x00data'
                # newcontent = header+content[68:]
                # actual content should start at 68, but for some reason, there is some data trash in the beginning of the file!? Skip that!
                newcontent = header+content[350:]
                newfile.write(newcontent)
                newfile.close()
                os.remove(backup)
            except:
                print("Error: Header repair failed!")

    def close(self):
        self.lib.vtlClose()

    def __del__(self):
        self.close()
