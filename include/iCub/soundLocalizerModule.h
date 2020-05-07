#ifndef SOUND_LOCALIZER_MODULE_H
#define SOUND_LOCALIZER_MODULE_H


//
/*
  * Copyright (C)2018  Department of Robotics Brain and Cognitive Sciences - Istituto Italiano di Tecnologia
  * Author: francesco rea, jonas gonzalez
  * email: jonas.gonzalez@iit.it
  * Permission is granted to copy, distribute, and/or modify this program
  * under the terms of the GNU General Public License, version 2 or any
  * later version published by the Free Software Foundation.
  *
  * A copy of the license can be found at
  * http://www.robotcub.org/icub/license/gpl.txt
  *
  * This program is distributed in the hope that it will be useful, but
  * WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
  * Public License for more details
*/
//



/**
 *
 * \defgroup
 * @ingroup
 *
 * Description Module
 *
 *
 *
 * \section lib_sec Libraries
 *
 * YARP.
 *
 * \section parameters_sec Parameters
 *
 * <b>Command-line Parameters</b>
 *
 * The following key-value pairs can be specified as command-line parameters by prefixing \c -- to the key
 * (e.g. \c --from file.ini. The value part can be changed to suit your needs; the default values are shown below.
 *
 * - \c from \c config.ini \n
 *   specifies the configuration file
 *
 * - \c context \c icub_testMOdule/conf \n
 *   specifies the sub-path from \c ICUB_ROOT/icub/app to the configuration file
 *
 * - \c name \c icub_interactionInterface \n
 *   specifies the name of the testMOduleThread (used to form the stem of testMOduleRateThread port names)
 *
 * - \c robot \c icub \n
 *   specifies the name of the robot (used to form the root of robot port names)
 *
 * - \c config  \n
 *   specifies the name of the script that will be used
 *
 *
 * <b>Configuration File Parameters</b>
 *
 * The following key-value pairs can be specified as parameters in the configuration file
 * (they can also be specified as command-line parameters if you so wish).
 * The value part can be changed to suit your needs; the default values are shown below.
 *
 *
 *
 * \section portsa_sec Ports Accessed
 *
 * - None
 *
 * \section portsc_sec Ports Created
 *
 *  <b>Input ports</b>
 *
 *  - \c /attentionInhibitorModule \n
 *    This port is used to send actions to execute to the specified script. \n
 *    The following commands are available
 *
 *  -  \c help \n
 *  -  \c quit \n
 *
 *
 *    Note that the name of this port mirrors whatever is provided by the \c --name parameter value
 *    The port is attached to the terminal so that you can type in commands and receive replies.
 *    The port can be used by other testMOduleRateThreads but also interactively by a user through the yarp rpc directive, viz.: \c yarp \c rpc \c /testMOdule
 *    This opens a connection from a terminal to the port and allows the user to then type in commands and receive replies.
 *
 *
 * <b>Output ports</b>
 *
 *  - None
 *
 * \section in_files_sec Input Data Files
 *
 * None
 *
 * \section out_data_sec Output Data Files
 *
 * None
 *
 * \section conf_file_sec Configuration Files
 *
 * \c attentionInhibitorModule.ini  in \c app/ \n
 *
 * \section tested_os_sec Tested OS
 *
 * Linux
 *
 * \section example_sec Example Instantiation of the Module
 *
 * <tt>attentionInhibitorModule --name attentionInhibitorModule </tt>
 *
 * \author jonas gonzalez
 *
 * Copyright (C) 2011 RobotCub Consortium\n
 * CopyPolicy: Released under the terms of the GNU GPL v2.0.\n
 * This file can be edited at \c $/src/
 *
 */



#include <iostream>
#include <string>


#include <yarp/os/RFModule.h>
#include <yarp/os/RpcClient.h>

#include <yarp/os/Network.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/os/Thread.h>
#include <yarp/os/Vocab.h>
#include <yarp/os/Semaphore.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Image.h>
#include <yarp/sig/Matrix.h>
#include <array>
#include <opencv/cv.h>
#include <opencv2/features2d.hpp>




class soundLocalizerModule : public yarp::os::RFModule {


public:
    /**
    *  configure all the attentionInhibitorModule parameters and return true if successful
    * @param rf reference to the resource finder
    * @return flag for the success
    */
    bool configure(yarp::os::ResourceFinder &rf) override;

    /**
    *  interrupt, e.g., the ports
    */
    bool interruptModule() override;

    /**-
    *  close and shut down
    */
    bool close() override;

    /**
    *  to respond through rpc port
    * @param command reference to bottle given to rpc port of module, alongwith parameters
    * @param reply reference to bottle returned by the rpc port in response to command
    * @return bool flag for the success of response else termination of module
    */
    bool respond(const yarp::os::Bottle &command, yarp::os::Bottle &reply) override;

    /**
    *  implemented to define the periodicity of the module
    */
    double getPeriod() override;

    /**
    *  thread run function
    */
    bool updateModule() override;


private :
    enum {
        // general command vocab's
        COMMAND_VOCAB_OK = yarp::os::createVocab('o', 'k'),

        COMMAND_VOCAB_SET = yarp::os::createVocab('s', 'e', 't'),
        COMMAND_VOCAB_GET = yarp::os::createVocab('g', 'e', 't'),
        COMMAND_VOCAB_SUSPEND = yarp::os::createVocab('s', 'u', 's'),
        COMMAND_VOCAB_RES = yarp::os::createVocab('r', 'e', 's'),
        COMMAND_VOCAB_DEL = yarp::os::createVocab('d', 'e', 'l'),
        COMMAND_VOCAB_ADD = yarp::os::createVocab('a', 'd', 'd'),

        COMMAND_VOCAB_HELP = yarp::os::createVocab('h', 'e', 'l', 'p'),
        COMMAND_VOCAB_FAILED = yarp::os::createVocab('f', 'a', 'i', 'l'),
        COMMAND_VOCAB_HIGH_THRESHOLD = yarp::os::createVocab('h', 't', 'h', 'r'),
        COMMAND_VOCAB_LOW_THRESHOLD = yarp::os::createVocab('l', 't', 'h', 'r'),
        COMMAND_VOCAB_DRAW = yarp::os::createVocab('d', 'r', 'a', 'w'),
        COMMAND_VOCAB_LEFT = yarp::os::createVocab('l', 't'),
        COMMAND_VOCAB_RIGHT = yarp::os::createVocab('r', 't')

    };


    double highProbabilityThreshold, lowProbabilityThreshold;

    std::string moduleName;                  // name of the module
    std::string robotName;                   // name of the robot
    std::string handlerPortName;             // name of handler port

    int width, height;

    bool drawOnLeft;
    bool drawOnRight;
    bool process;

    cv::Mat speakerImg, headImg, *template_img;

    /*  */
    yarp::os::Port handlerPort;              // a port to handle messages
    yarp::os::Semaphore mutex;                  // semaphore for the respond function
    yarp::os::BufferedPort<yarp::os::Bottle> soundRecorderClientRPC;
    yarp::os::RpcClient faceDetectorClientRpc;
    yarp::os::Port faceCoordinatePort;

    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb> > outputImagePort;
    yarp::os::BufferedPort<yarp::os::Bottle>  outputAnglePort;
    yarp::os::BufferedPort<yarp::sig::Matrix> anglePositionPort;


    /**
     * Compute if the current angle is at the left or right in allocentric reference frame
     * @param angles
     * @return {-1 : no relevant angle, 0 : left, 1 : right},
     */
    int computePositionAngle(yarp::sig::Matrix angles);

    /**
     * Execute a look command with iKinGazeCtrl
     * @param angle value from the egocentric reference
     * @return
     */
    bool lookAngle(const int& angle);


    /**
     * Draw a radar style grid on the img
     * @param img
     * @param scale
     * @param color
     */
    void drawGrid(cv::Mat img, double scale, CvScalar color);


    void drawLeft(cv::Mat &img);

    void drawRight(cv::Mat &img);

    void overlayImage(const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output, cv::Point2i location);

    void writeImage();

    bool openIkinGazeCtrl();

    //iKinGazeCtrl parameters
    int ikinGazeCtrl_Startcontext{}, gaze_context{};
    yarp::dev::PolyDriver *clientGaze{};
    yarp::dev::IGazeControl *iGaze{};
    bool enableSaccade, withFaceDetector;

    bool processFace(bool enable);

    void getCenterFace(const yarp::os::Bottle& coordinate, yarp::sig::Vector &pixelLoc);
};
#endif //SOUND_LOCALIZER_MODULE_H
