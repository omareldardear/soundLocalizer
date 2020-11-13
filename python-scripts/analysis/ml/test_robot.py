import tensorflow as tf

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from utils import get_fft_gram
import soundfile as sf

import scipy.io.wavfile as wavfile

import librosa
import sys
import yarp
import numpy as np
from CONFIG import *

import scipy


# subjects_dic = {0: 'NOEMIE', 1: 'BACKGROUND', 2: 'VALERIA', 3:'MELLY', 4: 'JONAS', 5: 'FABIO', 6: 'CARLO', 7: 'GIULIA',
#  8: 'MARCO', 9: 'FRANCESCO', 10: 'DARIO', 11: 'LINDA', 12: 'LUCA'}


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


yarpLog = yarp.Log()


class SpeakeRecognitionModule(yarp.RFModule):
    """
    Description:
        Class to recognize speaker from the audio

    Args:
        input_port  : Audio from remoteInterface

    """

    def __init__(self):
        yarp.RFModule.__init__(self)

        # handle port for the RFModule
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        # Define vars to receive an image
        self.audio_in_port = yarp.Port()


        self.head_motorPort = yarp.Port()


        self.module_name = None

        self.model_path = None

        self.sound = yarp.Sound()

        self.start_ts = -1
        self.model = None

        self.audio = []

        self.record = False

        self.np_audio = None

        self.length_input = 1


    def configure(self, rf):
        self.module_name = rf.check("name",
                                    yarp.Value("SpeakerRecognition"),
                                    "module name (string)").asString()

        self.model_path = rf.check("path",
                                    yarp.Value("/home/jonas/CLionProjects/soundLocalizer/python-scripts/analysis/data/saved_model/my_model.h5"),
                                    "Model path (.h5) (string)").asString()

        # Create handle port to read message
        self.handle_port.open('/' + self.module_name)

        # Create a port to receive an audio object
        self.audio_in_port.open('/' + self.module_name + '/recorder:i')

        print("TensorFlow version: {}".format(tf.__version__))

        # Create a new model instance
        self.model = tf.keras.models.load_model(self.model_path)

        self.head_motorPort.open('/' + self.module_name + 'angle:o')

        print("Model successfully loaded, running ")

        yarpLog.info("Initialization complete")

        return True

    def interruptModule(self):
        yarpLog.info("stopping the module")
        self.audio_in_port.interrupt()
        self.handle_port.interrupt()

        return True

    def close(self):
        self.audio_in_port.close()
        self.handle_port.close()

        return True

    def respond(self, command, reply):
        ok = False

        # Is the command recognized
        rec = False

        reply.clear()

        if command.get(0).asString() == "quit":
            reply.addString("quitting")
            return False



        return True

    def getPeriod(self):
        """
           Module refresh rate.

           Returns : The period of the module in seconds.
        """
        return 0.05

    def updateModule(self):

        if self.audio_in_port.getInputCount():
            self.audio_in_port.read(self.sound)

            chunk = np.zeros((self.sound.getChannels(), self.sound.getSamples()), dtype=np.float32)

            for c in range(self.sound.getChannels()):
                for i in range(self.sound.getSamples()):
                    chunk[c][i] = self.sound.get(i, c) / 32768.0

            self.audio.append(chunk)

            np_audio = np.concatenate(self.audio, axis=1)
            np_audio = librosa.util.normalize(np_audio, axis=1)
            np_audio = np.squeeze(np_audio)
            signal = np.transpose(np_audio, (1, 0))
            if len(signal) >= self.length_input * self.sound.getFrequency():

                # sf.write('/home/jonas/Desktop/test'+ str(self.i) +'.wav', signal, self.sound.getFrequency())


                angle_predicted = self.get_speaker(signal)

                if self.head_motorPort.getOutputCount():
                    head_bottle = yarp.Bottle()
                    head_bottle.clear()
                    head_bottle.addString("abs")
                    head_bottle.addDouble(angle_predicted)
                    head_bottle.addDouble(0.0)
                    head_bottle.addDouble(0.0)

                    self.head_motorPort.write(head_bottle)


        return True




    def get_speaker(self, signal):

        # fs, signal = wavfile.read('/home/jonas/Desktop/test'+ str(self.i) +'.wav', "wb")

        signal1 = signal[:, 0]
        signal2 = signal[:, 1]

        if RESAMPLING_F:
            signal1 = np.array(scipy.signal.resample(signal1, RESAMPLING_F), dtype=np.int16)
            signal2 = np.array(scipy.signal.resample(signal2, RESAMPLING_F), dtype=np.int16)


        fft_gram1, fft_gram2 = get_fft_gram(signal, self.sound.getFrequency())
        input_x = np.stack((fft_gram1, fft_gram2), axis=-1)
        input_x = np.expand_dims(input_x, axis=0)

        y_pred = self.model.predict(input_x)
        angle_pred = ((y_pred * 180))[0][0] - 90

        print(f"Predictions {angle_pred}")

        self.audio = []
        self.model.reset_states()

        return angle_pred



if __name__ == '__main__':

    # Initialise YARP
    if not yarp.Network.checkNetwork():
        print("Unable to find a yarp server exiting ...")
        sys.exit(1)

    yarp.Network.init()

    audioRecorderModule = SpeakeRecognitionModule()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('speakerRecognition')
    rf.setDefaultConfigFile('speakerRecognition.ini')

    if rf.configure(sys.argv):
        audioRecorderModule.runModule(rf)
    sys.exit()
