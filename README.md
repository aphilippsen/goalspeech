# GoALSpeech: Goal-directed Articulatory Learning for Speech Acquisition #

Source code for replicating the results of the following paper:
Philippsen, A. (2021). Goal-directed exploration for learning vowels and syllables: a computational model of speech acquisition. KI-Künstliche Intelligenz, 35(1), 53-70.

This the python implementation of the original Matlab implementation used for Ph.D. thesis on
"Learning How to Speak. Goal Space Exploration for Articulatory Skill Acquisition":
https://pub.uni-bielefeld.de/record/2921296

## Installation ##

The code runs with Python3.

### Python packages ###

Can be installed e.g. via pip:

* dtw (tested with version 1.4.0)
* matplotlib (you might need to install the package python3-tk)
* numpy
* python-speech-features (https://github.com/StevenLOL/python_speech_features)
* scipy
* scikit-learn
* sounddevice (you might need to install the package for the PortAudio library first in some Linux distributions, libportaudio2)
* torch
* oct2py (If GBFB features should be used, see below.)
* tqdm (progress bar for sound production)
* fastdtw (Comparison of sounds using the syllable weighting scheme)

### Articulatory system ###

* VocalTractLab (VTL):
Download VocalTractLab from vocaltractlab.de/index.php?page=vocaltractlab-download and unpack the contents into ./libs:
libs/VTL2.1_Linux    (http://vocaltractlab.de/download-vocaltractlabapi/VTL2.1_Linux.zip)
libs/vtlapi-2.1b     (http://vocaltractlab.de/download-vocaltractlabapi/vtlapi-2.1b.zip)

* Syllable gestures:
data/kroeger_ges contains VTL gesture files provided by Bernd Kröger on his website: http://www.phonetik.phoniatrie.rwth-aachen.de/bkroeger/research.htm

### Acoustics ###

* Praat:
Install praat or download the executable file from: http://www.fon.hum.uva.nl/praat/ (* https://www.fon.hum.uva.nl/praat/praat6108_linux64.tar.gz). Define where to locate it on your computer via an environment variable "PRAAT_CMD", e.g. by adding the following to your ~/.bashrc:
  export PRAAT_CMD=/usr/bin/praat

* GBFB features:
For using the Gabor filterbank (GBFB) features, the following github project is required to lie in libs/
https://github.com/m-r-s/reference-feature-extraction
Furthermore, octave has to be installed on the system and oct2py is required as a Python package.

## How to run ##

*1. Configure the parameters.*
The parameters for an experiment are defined in a cfg file. Examples can be found in goalspeech/config/. Details about the format can be found in goalspeech/config/info.txt.
A config file can also be generated via the file generateConfig.py. Modify the script and execute it to generate the file which will be written to config/.

*2. Initialize the experiment.*
In ipython:
Run one of the files initExperiment*.py [e.g. "%run -i initExperimentVowels.py"]. If you changed the config and want to use your own one, replace the path for the config file first.
This will create an instance of VTLSpeechProduction and loads all the required parameters from the config into the ipython workspace.
(When switching between *Vowels.py and *Syllables.py, currently a new ipython instance has to be started.)

*3. Run experiment.*
After the initialization, runExperiment*.py starts the babbling learning process.
If the script is run for the first time, ambient speech data is generated and stored into data/ambientSpeech/. In subsequent runs this file is reused. If you want the system to override it, delete it in the above mentioned directory.
The following steps are performed:
  * Create articulatory data set (temporarily, is discarded after generating the acoustics)
  * Create corresponding acoustic data set, store in data/ambientSpeech (used as ambient speech, i.e. the speech from the environment that the system hears in its environment)
  * Start babbling. Results will be saved into a folder named with the current date in data/results/.

*4. Inspect results.*
* In the beginning of babbling: gs.png shows the generated goal space. Make sure that it looks meaningful before investing time in continuing the babbling. The config is stored as config.txt.
* After each babbling run (#runs defined in "runs") the results of the corresponding run R are stored as "results-R.pickle"
* The script evaluateResults.py can be used to evaluate and visualize the results from multiple runs of one experiment.
