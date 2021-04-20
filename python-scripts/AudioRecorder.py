import threading

import librosa

import sys
import os
import time
import yarp
import numpy as np
import soundfile as sf

yarpLog = yarp.Log()


class AudioRecorderModule(yarp.RFModule):
    """
    Description:
        Class to record audio form the iCub head

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

        self.module_name = None

        self.saving_path = None

        self.sound = yarp.Sound()

        self.start_ts = -1
        self.date_path = time.strftime("%Y%m%d-%H%M%S")

        self.audio = []

        self.record = False

        self.np_audio = None

        self.stop_ts = -1

    def configure(self, rf):
        self.module_name = rf.check("name",
                                    yarp.Value("AudioRecorder"),
                                    "module name (string)").asString()

        self.saving_path = rf.check("path",
                                    yarp.Value("/tmp"),
                                    "saving path name (string)").asString()

        # Create handle port to read message
        self.handle_port.open('/' + self.module_name)

        # Create a port to receive an audio object
        self.audio_in_port.open('/' + self.module_name + '/recorder:i')

        if not os.path.exists(f'{self.saving_path}/{self.date_path}'):
            yarpLog.info("Creating directory for saving")
            os.makedirs(f'{self.saving_path}/{self.date_path}')

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

        elif command.get(0).asString() == "help":
            reply.addVocab(yarp.encode("many"))
            reply.addString("AudioRecorder module commands are")

            reply.addString("start : Start the recording")
            reply.addString("stop : Stop the recording")
            reply.addString("save : Stop and save the recording")
            reply.addString("drop : Drop the recording")

        elif command.get(0).asString() == "start":
            if self.audio_in_port.getInputCount():
                self.audio = []
                self.record = True
                self.start_ts = time.time()

                yarpLog.info("starting recording!")

                reply.addString("ok")
            else:
                reply.addString("nack")

        elif command.get(0).asString() == "stop":
            yarpLog.info("stopping recording!")
            self.record = False
            self.stop_ts = time.time()

            reply.addString("ok")

        elif command.get(0).asString() == "drop":
            yarpLog.info("dropping recording!")
            self.record = False
            reply.addString("ok")

        elif command.get(0).asString() == "save":
            yarpLog.info("Saving recording!")
            self.record = False
            self.stop_ts = time.time()
            self.save_recording()
            filePath = f'{self.saving_path}/{self.date_path}/'
            fileName = f'{self.start_ts}_{self.stop_ts}.wav'
            reply.addString("ok")
            reply.addString(filePath)
            reply.addString(fileName)
        return True

    def getPeriod(self):
        """
           Module refresh rate.

           Returns : The period of the module in seconds.
        """
        return 0.05

    def updateModule(self):

        if self.record:

            self.audio_in_port.read(self.sound)

            chunk = np.zeros((self.sound.getChannels(), self.sound.getSamples()), dtype=np.float32)

            for c in range(self.sound.getChannels()):
                for i in range(self.sound.getSamples()):
                    chunk[c][i] = self.sound.get(i, c) / 32768.0

            self.audio.append(chunk)
        return True

    def save_recording(self):

        np_audio = np.concatenate(self.audio, axis=1)
        np_audio = librosa.util.normalize(np_audio, axis=1)
        filename = f'{self.saving_path}/{self.date_path}/{self.start_ts}_{self.stop_ts}.wav'
        np_audio = np.squeeze(np_audio)
        a = np.transpose(np_audio, (1, 0))
        sf.write(filename, a, self.sound.getFrequency())


if __name__ == '__main__':

    # Initialise YARP
    if not yarp.Network.checkNetwork():
        print("Unable to find a yarp server exiting ...")
        sys.exit(1)

    yarp.Network.init()

    audioRecorderModule = AudioRecorderModule()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('audioRecorder')
    rf.setDefaultConfigFile('audioRecorder.ini')

    if rf.configure(sys.argv):
        audioRecorderModule.runModule(rf)
    sys.exit()
