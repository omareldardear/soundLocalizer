import threading

import librosa

import sys
import os
import time
import yarp
import numpy as np
import soundfile as sf

yarpLog = yarp.Log()


class ObjectDetectorModule(yarp.RFModule):
    """
    Description:
        Object to read yarp image and localise and recognize objects

    Args:
        input_port  : input port of image
        output_port : output port for streaming recognized names
        display_port: output port for image with recognized objects in bouding box
        raw_output : output the list of <bounding_box, label, probability> detected objects
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
                                    yarp.Value("audioRecorder"),
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

            yarpLog.info("saved recording!")

            reply.addString("ok")

        elif command.get(0).asString() == "drop":
            yarpLog.info("dropping recording!")
            self.record = False
            reply.addString("ok")

        elif command.get(0).asString() == "save":
            yarpLog.info("Saving recording!")
            self.save_recording()

            reply.addString("ok")

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
        self.record = False

        np_audio = np.concatenate(self.audio, axis=1)
        np_audio = librosa.util.normalize(np_audio, axis=1)

        sf.write(f'{self.saving_path}/{self.date_path}/{self.start_ts}_{self.stop_ts}.wav', np.squeeze(np_audio),
                 self.sound.getFrequency())



if __name__ == '__main__':

    # Initialise YARP
    if not yarp.Network.checkNetwork():
        print("Unable to find a yarp server exiting ...")
        sys.exit(1)

    yarp.Network.init()

    objectsDetectorModule = ObjectDetectorModule()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('audioRecorder')
    rf.setDefaultConfigFile('audioRecorder.ini')

    if rf.configure(sys.argv):
        objectsDetectorModule.runModule(rf)
    sys.exit()
