import threading

import librosa

import os
import time
import yarp
import numpy as np
import soundfile as sf
import argparse

# Init Yarp.
yarp.Network.init()


def get_args():
    parser = argparse.ArgumentParser(description='rpc test')
    parser.add_argument('-n', '--name', default="/audioRecorder", help='Name for the module. (default: audioRecorder)')
    parser.add_argument('-s', '--save',    default='/tmp', help='dest dir')
    parser.add_argument('-f', '--num_frames', default=40, help='number of frames')
    parser.add_argument('-i', '--raw_audio_port', default="/rawAudio:o", help='input audio port')


    args = parser.parse_args()
    return args


class AudioRecorder(object):
    def __init__(self, args):

        # Init the Yarp Port.
        self.port_name = args.name

        self.input_port = yarp.RpcServer()
        self.input_port.open(self.port_name + "/rpc:i")

        self.target_dir = args.save

        self.audio_in_port = yarp.Port()
        self.audio_in_port.open(self.port_name + "/recorder")

        yarp.Network.connect(args.raw_audio_port, self.port_name + "/recorder")

        self.sound = yarp.Sound()

        self.count = 0
        self.date_path = time.strftime("%Y%m%d-%H%M%S")
        self.num_frames = args.num_frames
        self.audio = []

        if not os.path.exists(f'{self.target_dir}/{self.date_path}'):
            os.makedirs(f'{self.target_dir}/{self.date_path}')

    def run(self):

        # Initialize yarp containers
        command = yarp.Bottle()
        reply = yarp.Bottle()

        # Loop until told to quit.
        while True:
            self.input_port.read(command, True)
            reply.clear()

            cmd = command.get(0).asString()
            len = command.size()

            # Kill this script.
            if cmd == "quit" or cmd == "--quit" or cmd == "-q" or cmd == "-Q":
                reply.addString("Good Bye!")
                # self.input_port.reply(reply) # Don't return yet.
                # return

            # Get a help Message.
            elif cmd == "help" or cmd == "--help" or cmd == "-h" or cmd == "-H":
                reply.addString(  # Please don't look at this.
                    "Commands: start, stop. ping, quit"
                )

            # Ping the server.
            elif cmd == "ping" or cmd == "--ping":
                reply.addString("Oll Korrect")

            elif cmd == "start" or cmd == "--start":
                self.t = threading.Thread(name='child procs', target=self.start_recording)
                print("starting thread!")
                self.t.start()
                print("started thread!")
                reply.addString("ok")

            elif cmd == "stop" or cmd == "--stop":
                self.stop_recording()
                reply.addString(f"ok")

            # Catch Unknown Commands.
            elif len == 0:
                continue
            else:
                reply.addString("Unkown")
                reply.addString("Command")
                reply.append(command)

            # Reply back on the rpc port.
            self.input_port.reply(reply)

            # Leave loop if requested.
            if reply.get(0).asString() == "Good Bye!":
                return

    def start_recording(self):
        self.record = True

        self.audio = []

        while self.record:

            self.audio_in_port.read(self.sound)

            chunk = np.zeros((self.sound.getChannels(), self.sound.getSamples()), dtype=np.float32)

            for c in range(self.sound.getChannels()):
                for i in range(self.sound.getSamples()):
                    chunk[c][i] = self.sound.get(i, c)/32768.0

            self.audio.append(chunk)

    def stop_recording(self):
        self.record = False
        self.np_audio = np.concatenate(self.audio, axis=1)
        self.np_audio = librosa.util.normalize(self.np_audio, axis=1)

        print(self.np_audio.shape)
        print(self.sound.getFrequency())

        timestamp =  time.time()

        sf.write(f'{self.target_dir}/{self.date_path}/{self.count}_{timestamp}.wav', np.squeeze(self.np_audio[0,:]), self.sound.getFrequency())
        self.count += 1

    def cleanup(self):
        print("Closing RPC Server.")
        self.input_port.close()

def main():
    args = get_args()

    rp = AudioRecorder(args)

    try:
        rp.run()
    finally:
        rp.cleanup()


if __name__ == '__main__':
    main()
