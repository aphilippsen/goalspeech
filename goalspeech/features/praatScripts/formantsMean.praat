# Praat Script for Formant Analysis
###################################

form Formant extraction
    sentence Wav_file test.wav
    word speaker male
endform

if wav_file$ = ""
    printline missing file name!
    exit
endif

if speaker$ = ""
    printline missing speaker info!
    exit
endif

# Read in File
Read from file... 'wav_file$'


# time step; maximum number of formants; maximum formant;window length; pre-emphasis
# max formant: 5000 Hz (male adult), 5500 Hz (female adult); ~8000 Hz (young child)
# Pre-emphasis from (Hz): the +3 dB point for an inverted low-pass filter with a slope of +6 dB/octave. If this value is 50 Hz, then frequencies below 50 Hz are not enhanced, frequencies around 100 Hz are amplified by 6 dB, frequencies around 200 Hz are amplified by 12 dB, and so forth. The point of this is that vowel spectra tend to fall by 6 dB per octave; the pre-emphasis creates a flatter spectrum, which is better for formant analysis because we want our formants to match the local peaks, not the global spectral slope.

female = 5500
# suggestion: child = 8000
# suggestion: male = 5000

# for vtl:
male = 4000
male2 = 4500
maleTest = 4500
# male = 5000
child = 8000

if speaker$ = "male"
    maxFormant = male
elsif speaker$ = "male2"
    maxFormant = male2
elsif speaker$ = "maleTest"
    maxFormant = maleTest
elsif speaker$ = "female"
    maxFormant = female
elsif speaker$ = "child"
    maxFormant = child
endif


To Formant (burg)... 0.0 5 maxFormant 0.025 50


# Put Mean(F0) to a variable
f1value = Get mean... 1 0.0 0.0 Hertz
f2value = Get mean... 2 0.0 0.0 Hertz
f3value = Get mean... 3 0.0 0.0 Hertz

# Print result to command window
printline 'f1value:2' 'f2value:2' 'f3value:2'

###################################
