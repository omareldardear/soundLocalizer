# soundLocalizer
Sound localizer and data dumper for experiment with iCub


## Experiment setup
### Application dependencies
This the application that need to run on the iCub setup

    Audio_Attention
    Face_Duo_Expression
    iKinGazeCtrl
    Left Camera

### Sound Localizer Modules
1. Run every modules 
2. Connect everything except the /souundLocalizer/angle:i  and the /logHeadState connection


### Experiment methodology
Explain to  people that they will need to attract iCub attention by speaking to him. When the iCub successfully
found the sources speech it will smile. This is the signal that the participant can move to another location.
    
    Note: You need to explain carrefully that they  need to continuously speaks to the iCub

To start the experiment connect the /souundLocalizer/angle:i  and the /logHeadState connection

### Data recording
There is two data_dumpers (logAngles, logHeadState) and they will create the lof file in the /home folder. 
The audio recording by defaults is saved into tmp/<date_of_the_day>.

### End of experiment
1. Stop the soundLocalizer and AudioRecorder and the  yarpdatadumpers
2. Copy the data and clean the /home and /tmp folders


## SoundLocalizer Parameters
High Probability

    Define the upper threshold for the sound detection
    
Low Probability

    Define the lower threshold to reset the detection
    
Both parameters can be modified/read trough the rpc port /soundLocalizer

    High threshold : set/get hthr <double_value> 
    Low threshold : set/get lthr <double_value> 