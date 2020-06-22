#include <yarp/os/Log.h>
//
#include <vector>
#include <yarp/cv/Cv.h>

#include <opencv2/highgui.hpp>
#include <iCub/soundLocalizerModule.h>

/*
  * Copyright (C)2017  Department of Robotics Brain and Cognitive Sciences - Istituto Italiano di Tecnologia
  * Author: jonas gonzalez
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




const CvScalar color_bwhite = cvScalar(200, 200, 255);
const CvScalar color_white = cvScalar(255, 255, 255);
const CvScalar color_red = cvScalar(0, 0, 255);
const CvScalar color_yellow = cvScalar(0, 255, 255);
const CvScalar color_black = cvScalar(0, 0, 0);
const CvScalar color_gray = cvScalar(100, 100, 100);
const CvScalar color_green = cvScalar(0, 242, 0);
const CvScalar color_blue = cvScalar(230, 0, 0);

/*
 * Configure method. Receive a previously initialized
 * resource finder object. Use it to configure your module.
 * If you are migrating from the old Module, this is the
 *  equivalent of the "open" method.
 */

bool soundLocalizerModule::configure(yarp::os::ResourceFinder &rf) {
    using namespace yarp::os;
    using namespace yarp::sig;

    if (rf.check("help")) {
        printf("HELP \n");
        printf("====== \n");
        printf("--name           : changes the rootname of the module ports \n");
        printf("--robot          : changes the name of the robot where the module interfaces to  \n");
        printf("--name           : rootname for all the connection of the module \n");
        printf("--probabilityThreshold       : probability threshold to trigger the motion \n");
        printf(" \n");
        printf("press CTRL-C to stop... \n");
        return true;
    }

    highProbabilityThreshold = rf.check("probabilityThresholdHigh",
                                        Value(0.0042),
                                        "high probability threshold to trigger the motion (double)").asDouble();

    lowProbabilityThreshold = rf.check("probabilityThresholdLow",
                                       Value(0.0035),
                                       "low probability threshold to trigger the motion (double)").asDouble();

    process = true;

    /* get the module name which will form the stem of all module port names */
    moduleName = rf.check("name",
                          Value("/soundLocalizer"),
                          "module name (string)").asString();
    /*
    * before continuing, set the module name before getting any other parameters,
    * specifically the port names which are dependent on the module name
    */
    setName(moduleName.c_str());

    /*
    * get the robot name which will form the stem of the robot ports names
    * and append the specific part and device required
    */
    robotName = rf.check("robot",
                         Value("icub"),
                         "Robot name (string)").asString();

    /*
    * attach a port of the same name as the module (prefixed with a /) to the module
    * so that messages received from the port are redirected to the respond method
    */
    handlerPortName = "";
    handlerPortName += getName();         // use getName() rather than a literal




    if (!handlerPort.open(handlerPortName)) {
        yInfo(" Unable to open port %s", this->handlerPortName.c_str());
        return false;
    }

    attach(handlerPort);                  // attach to port

    if (!anglePositionPort.open(getName("/angle:i"))) {
        yInfo("Unable to open port /anglePositionPort");
        return false;  // unable to open; let RFModule know so that it won't run
    }


    if (!outputImagePort.open(getName("/image:o"))) {
        yInfo("Unable to open port /image:o");
        return false;  // unable to open; let RFModule know so that it won't run
    }

    if (!soundRecorderClientRPC.open(getName("/soundRecorderCmd:o"))) {
        yInfo("Unable to open port /soundRecorderCmd:o");
        return false;  // unable to open; let RFModule know so that it won't run
    }

    if (!faceDetectorClientRpc.open(getName("/faceDetectorRpc:o"))) {
        yInfo("Unable to open port /faceDetectorRpc:o");
        return false;  // unable to open; let RFModule know so that it won't run
    }


    if (!faceCoordinatePort.open(getName("/faceCoordinate:i"))) {
        yInfo("Unable to open port /faceCoordinate:i");
        return false;  // unable to open; let RFModule know so that it won't run
    }

    if (!outputAnglePort.open(getName("/outputAnglePort:o"))) {
        yInfo("Unable to open port /outputAnglePort:o");
        return false;  // unable to open; let RFModule know so that it won't run
    }

    if (!clientRPCEmotion.open(getName("/faceExpressions/cmd:o"))){
        yInfo("Unable to open port /faceExpressions/cmd:o");
        return false;
    }

    if (!NetworkBase::connect(clientRPCEmotion.getName(), "/icub/face/emotions/in")) {
        yInfo("Unable to connect to faceEmotion RPC ");
        return false;
    }


    width = rf.check("width",
                     Value(320),
                     "width of the image used for attention (int)").asInt();

    height = rf.check("height",
                      Value(240),
                      "height of the image used for attention (int)").asInt();


    const std::string images_path = rf.check("img_path",
                                             Value("/usr/local/src/robot/cognitiveInteraction/soundLocalizer/app"),
                                             "Path to the images (speaker, iCubHead) (string)").asString();

    if (images_path.empty()) {
        yError("You need to specify a valid path to load the speaker, iCubHead images");
        return false;

    }


    drawOnLeft = drawOnRight = enableAudioRecording = false;

    template_img = new cv::Mat(height, width, CV_8UC3, color_white);

    speakerImg = cv::imread(images_path + "/speaker_on.png", cv::IMREAD_UNCHANGED);
    cv::resize(speakerImg, speakerImg, cv::Size(40, 40));

    headImg = cv::imread(images_path + "/head.png", cv::IMREAD_UNCHANGED);
    cv::resize(headImg, headImg, cv::Size(40, 40));

    timeOut = rf.check("timeout", Value(5), "timeout threshold in seconds").asInt();


    if (!openIkinGazeCtrl()) {
        yError("Unable to open iKinGazeCtrl");
        return false;
    }

    enableAudioRecording = true;

    if (!NetworkBase::connect(faceDetectorClientRpc.getName(), "/ObjectsDetector")) {
        yInfo("Unable to connect to /ObjectsDetector check that ObjectsDetector is running");
        return false;
    }


    if (!NetworkBase::connect(soundRecorderClientRPC.getName(), "/audioRecorder")) {
        yInfo("Unable to connect to /audioRecorder check that AudioRecorder is running");
    }

    currentTime = timeSystem.nowSystem();

    yInfo("Initialization done");

    return true;       // let the RFModule know everything went well
    // so that it will then run the module
}

bool soundLocalizerModule::close() {
    handlerPort.close();
    anglePositionPort.close();
    outputImagePort.close();
    faceCoordinatePort.close();
    faceDetectorClientRpc.close();
    clientRPCEmotion.close();
    soundRecorderClientRPC.close();
    outputAnglePort.close();

    clientGaze->close();

    /* stop the thread */
    yInfo("Stopping the thread ");

    delete template_img;
    delete clientGaze;
    return true;
}

bool soundLocalizerModule::interruptModule() {
    yInfo("Interrupting the thread \n");
    lookAngle(150);

    saveAudio("drop");
    iGaze->restoreContext(gaze_context);

    faceCoordinatePort.interrupt();
    faceDetectorClientRpc.interrupt();
    handlerPort.interrupt();
    soundRecorderClientRPC.interrupt();
    anglePositionPort.interrupt();
    outputImagePort.interrupt();
    clientRPCEmotion.interrupt();
    outputAnglePort.interrupt();



    return true;
}


double soundLocalizerModule::getPeriod() {
    /* module periodicity (seconds), called implicitly by myModule */
    return 0.08;
}

bool soundLocalizerModule::respond(const yarp::os::Bottle &command, yarp::os::Bottle &reply) {
    using namespace yarp::os;
    using namespace yarp::sig;

    std::vector<std::string> replyScript;
    std::string helpMessage = std::string(getName()) +
                              " commands are: \n" +
                              "help \n" +
                              "quit \n";
    reply.clear();

    if (command.get(0).asString() == "quit") {
        reply.addString("quitting");
        return false;
    }

    bool ok = false;
    bool rec = false; // is the command recognized?

    mutex.wait();

    switch (command.get(0).asVocab()) {
        case COMMAND_VOCAB_HELP:
            rec = true;
            {
                reply.addVocab(Vocab::encode("many"));
                reply.addString("Command available : ");
                reply.addString("To get or set the high threshold detection : set/get hthr <value>");
                reply.addString("To get or set the low threshold detection : set/get lthr <value>");
                ok = true;
            }
            break;

        case COMMAND_VOCAB_SET:
            rec = true;
            {
                switch (command.get(1).asVocab()) {
                    case COMMAND_VOCAB_HIGH_THRESHOLD: {
                        ok = true;
                        const double new_threshold = command.get(2).asDouble();
                        highProbabilityThreshold = new_threshold > 0 ? new_threshold : highProbabilityThreshold;
                        break;
                    }

                    case COMMAND_VOCAB_LOW_THRESHOLD: {
                        ok = true;
                        const double new_threshold = command.get(2).asDouble();
                        lowProbabilityThreshold = new_threshold > 0 ? new_threshold : lowProbabilityThreshold;
                        break;
                    }
                    default:
                        ok = true;
                        yInfo("received an unknown request after SET");
                        break;
                }
            }
            break;

        case COMMAND_VOCAB_ADD:
            rec = true;
            {
                switch (command.get(1).asVocab()) {


                    default:
                        yInfo("received an unknown request after ADD");
                        break;
                }
            }
            break;

        case COMMAND_VOCAB_DEL:
            rec = true;
            {
                yInfo("Resetting grid");
                drawOnRight = false;
                drawOnLeft = false;
                ok = true;


            }
            break;

        case COMMAND_VOCAB_GET:
            rec = true;
            {
                switch (command.get(1).asVocab()) {
                    case COMMAND_VOCAB_HIGH_THRESHOLD: {
                        ok = true;
                        reply.addDouble(highProbabilityThreshold);
                        break;
                    }
                    case COMMAND_VOCAB_LOW_THRESHOLD: {
                        ok = true;
                        reply.addDouble(lowProbabilityThreshold);
                        break;
                    }
                    default:
                        yInfo("received an unknown request after a GET");
                        break;
                }
            }
            break;

        case COMMAND_VOCAB_SUSPEND:
            rec = true;
            {
                ok = true;
            }
            break;

        case COMMAND_VOCAB_RES:
            rec = true;
            {
                ok = true;
            }
            break;

        case COMMAND_VOCAB_DRAW:
            rec = true;
            {
                switch (command.get(1).asVocab()) {


                    case COMMAND_VOCAB_LEFT: {
                        drawLeft(*template_img);

                        break;
                    }

                    case COMMAND_VOCAB_RIGHT: {
                        drawRight(*template_img);

                        break;
                    }
                    default:
                        yInfo("received an unknown request after a GET");
                        break;
                }
                ok = true;
            }
            break;

        default:
            break;

    }
    mutex.post();

    if (!rec)
        ok = RFModule::respond(command, reply);

    if (!ok) {
        reply.clear();
        reply.addVocab(COMMAND_VOCAB_FAILED);
    } else
        reply.addVocab(COMMAND_VOCAB_OK);

    return ok;

}

/* Called periodically every getPeriod() seconds */
bool soundLocalizerModule::updateModule() {
    if(this->isStopping()){ return false;}

    template_img = new cv::Mat(height, width, CV_8UC3, color_white);

    drawGrid(*template_img, 80.0, color_blue);
    overlayImage(*template_img, headImg, *template_img,
                 cv::Point(width / 2 - headImg.rows / 2, height / 2 - headImg.cols / 2));

    yarp::sig::Matrix *inputBottle = anglePositionPort.read(false);
    yarp::os::Bottle *coordinate_face;

    timeDiff = timeSystem.nowSystem() - currentTime;




    if (inputBottle != nullptr && faceCoordinatePort.getInputCount()) {
        sendEmotion("all", "neu");

        if(timeDiff >= timeOut && !process){
            yDebug("Reached timeout reset processing");
            process = true;
        }
        if (soundRecorderClientRPC.getOutputCount() && enableAudioRecording) {
            saveAudio("start");
            enableAudioRecording = false;

        }

        const int res_angle = computePositionAngle(*inputBottle);

        if (res_angle > -1 && process) {

            process = false;
            saveAudio("stop");

            if (lookAngle(res_angle)) {
                // flush the port
                while(faceCoordinatePort.getPendingReads() > 0){
                    faceCoordinatePort.read(false);

                }
                timeSystem.delay(2);

                coordinate_face = faceCoordinatePort.read(false);

                if (coordinate_face != nullptr) {
                    // Read the face coordinate returned by the face detector module
                    yarp::os::Bottle *bottle_coord = coordinate_face->get(0).asList();
                    bottle_coord = bottle_coord->get(2).asList();
                    yarp::sig::Vector px(3);   // specify the pixel where to look

                    // Compute the center of the face in the image reference frame and gaze at it
                    getCenterFace(*bottle_coord, px);
                    // param 1 select the image plane: 0 (left), 1 (right)
                    // param 3 distance in meter
                    this->iGaze->lookAtFixationPoint(px);    // look!
                    this->iGaze->waitMotionDone(0.1, 1);

                    yInfo("Gazing at face");
                    sendEmotion("all", "hap");
                    timeSystem.delay(2);


                    // Get the current fixation of the robot point in angles and send it
                    yarp::sig::Vector fixationAngles;
                    this->iGaze->getAngles(fixationAngles);

                    if (outputAnglePort.getOutputCount()) {
                        yarp::os::Bottle &soundAngle = outputAnglePort.prepare();
                        soundAngle.clear();
                        soundAngle.addDouble(fixationAngles[0]);
                        soundAngle.addDouble(fixationAngles[1]);
                        soundAngle.addDouble(fixationAngles[2]);
                        outputAnglePort.write();


                    }

                    if (soundRecorderClientRPC.getOutputCount()) {
                        saveAudio("save");
                        saveAudio("start");

                    }
                    process=true;
                    timeSystem.delay(2);

                }

                else{
                    yInfo("No face found");
                    saveAudio("drop");
                    lookAngle(150);
                    enableAudioRecording=true;

                }

                currentTime = timeSystem.now();



            }
            else{
                yInfo("Angle out of range no motion");
                process = true;
            }

        }
        writeImage();
    }


    return true;
}


/****************************************************** PROCESSING *************************************************/

bool soundLocalizerModule::openIkinGazeCtrl() {


    //---------------------------------------------------------------
    yInfo("Opening the connection to the iKinGaze");
    yarp::os::Property optGaze; //("(device gazecontrollerclient)");
    optGaze.put("device", "gazecontrollerclient");
    optGaze.put("remote", "/iKinGazeCtrl");
    optGaze.put("local", "/soundLocalizer/gaze");

    clientGaze = new yarp::dev::PolyDriver();
    clientGaze->open(optGaze);
    iGaze = nullptr;
    yInfo("Connecting to the iKinGaze");
    if (!clientGaze->isValid()) {
        return false;
    }

    clientGaze->view(iGaze);
    iGaze->storeContext(&ikinGazeCtrl_Startcontext);


    //Set trajectory time:
    iGaze->blockNeckRoll(0.0);
    iGaze->clearNeckPitch();
//    iGaze->blockEyes(0.0);

//    iGaze->setNeckTrajTime(0.5);
//    iGaze->setEyesTrajTime(0.2);
//    iGaze->setTrackingMode(true);
//    iGaze->setVORGain(1.3);
//    iGaze->setOCRGain(0.7);

//    iGaze->storeContext(&gaze_context);

    yInfo("Initialization of iKingazeCtrl completed");
    return true;
}

bool soundLocalizerModule::saveAudio(std::string cmd) {
    yarp::os::Bottle cmdBottle, reply;

    cmdBottle.addString(cmd);
    soundRecorderClientRPC.write(cmdBottle, reply);

    return reply.get(0).asString() == "ok";
}

void soundLocalizerModule::getCenterFace(const yarp::os::Bottle &coordinate, yarp::sig::Vector &pixelLoc) {

    const int center_x = coordinate.get(0).asInt() + ((coordinate.get(2).asInt() - coordinate.get(0).asInt()) / 2);
    const int center_y = coordinate.get(1).asInt() + ((coordinate.get(3).asInt() - coordinate.get(1).asInt()) / 2);



    yarp::sig::Vector imageFramePosition(2);
    yarp::sig::Vector rootFramePosition(3);

    imageFramePosition[0] = center_x;
    imageFramePosition[1] = center_y;

    // On the 3D frame reference of the robot the X axis is the depth
    iGaze->get3DPoint(0, imageFramePosition, 1.0, pixelLoc);


    yInfo("Face found, center is %d %d", center_x, center_y);

}

int soundLocalizerModule::computePositionAngle(yarp::sig::Matrix angle_matrix) {

    const int maxAngle = 270;//(int) angle_matrix.cols();
    double saliencyValue;
    double maxSaliency = 0;
    int maxIndex = 0;
    int angle = 0;


    // Start wth index 180 as it is the zero angle
    for (int i = 90; i < maxAngle; ++i) {
        saliencyValue = angle_matrix[0][i];

        if (saliencyValue > maxSaliency) {
            maxSaliency = saliencyValue;
            maxIndex = i;
            angle = maxIndex;
        }
    }

    yDebug("Found valid angle %d with saliency %f Process is %d", angle, maxSaliency, process);


    if (maxSaliency <= lowProbabilityThreshold && !process ) {
        process = true;
    }

    if (maxSaliency >= highProbabilityThreshold){

        return angle;
    }

    return -1;
}

bool soundLocalizerModule::lookAngle(const int &angle) {

    yarp::sig::Vector ang(3);

    // Right source
    if ( angle < 144) {

        ang[0] = +80.0;                   // azimuth-component [deg]
        ang[1] = +0.0;                   // elevation-component [deg]
        ang[2] = +0.5;                   // vergence-component [deg]
        drawOnRight = true;
        drawOnLeft = false;
    }

        // Center source
    else if (angle >= 144 && angle < 250) {
        ang[0] = 0.0;                   // azimuth-component [deg]
        ang[1] = 0.0;                   // elevation-component [deg]
        ang[2] = 0.5;                   // vergence-component [deg]
        drawOnRight = false;
        drawOnLeft = false;
    }

        // Left source
    else if (angle >= 250 ) {
        ang[0] = -50.0;                   // azimuth-component [deg]
        ang[1] = 0.0;                   // elevation-component [deg]
        ang[2] = 0.5;                   // vergence-component [deg]
        drawOnRight = false;
        drawOnLeft = true;
    } else {
        ang[0] = 0.0;                   // azimuth-component [deg]
        ang[1] = 0.0;                   // elevation-component [deg]
        ang[2] = 0.5;
        drawOnRight = false;
        drawOnLeft = false;
        return false;
    }

    this->iGaze->lookAtAbsAngles(ang);
    this->iGaze->waitMotionDone(0.1, 1);


    return true;
}



/****************************************************** VISUALIZATION *************************************************/

void soundLocalizerModule::drawGrid(cv::Mat img, double scale, CvScalar color) {
    // Vertical horizontal lines
    cv::line(img, cv::Point(0, 0), cv::Point(img.cols, img.rows), color);
    cv::line(img, cv::Point(img.cols, 0), cv::Point(0, img.rows), color);

    // Diagonale lines
    cv::line(img, cv::Point(img.cols / 2, 0), cv::Point(img.cols / 2, img.rows), color);
    cv::line(img, cv::Point(0, img.rows / 2), cv::Point(img.cols, img.rows / 2), color);
    const auto step = (int) (0.5 * scale); //mm

    char buff[10];
    int rad_step = 0;
    if (scale > 60)
        rad_step = 1;
    else
        rad_step = 2;
    for (int rad = 0; rad < 10; rad += rad_step) {
        sprintf(buff, "%3.1fm", float(rad) / 2);
        cv::circle(img, cv::Point(img.cols / 2, img.rows / 2), step * rad, color);
    }

}

void soundLocalizerModule::overlayImage(const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output,
                                        cv::Point2i location) {
    // taken from http://jepsonsblog.blogspot.com/2012/10/overlay-transparent-image-in-opencv.html
    background.copyTo(output);

    // start at the row indicated by location, or at row 0 if location.y is negative.
    for (int y = std::max(location.y, 0); y < background.rows; ++y) {
        int fY = y - location.y; // because of the translation

        // we are done of we have processed all rows of the foreground image.
        if (fY >= foreground.rows)
            break;

        // start at the column indicated by location,

        // or at column 0 if location.x is negative.
        for (int x = std::max(location.x, 0); x < background.cols; ++x) {
            int fX = x - location.x; // because of the translation.

            // we are done with this row if the column is outside of the foreground image.
            if (fX >= foreground.cols)
                break;

            // determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
            double opacity =
                    ((double) foreground.data[fY * foreground.step + fX * foreground.channels() + 3])

                    / 255.;

            // and now combine the background and foreground pixel, using the opacity,

            // but only if opacity > 0.
            for (int c = 0; opacity > 0 && c < output.channels(); ++c) {
                unsigned char foregroundPx =
                        foreground.data[fY * foreground.step + fX * foreground.channels() + c];
                unsigned char backgroundPx =
                        background.data[y * background.step + x * background.channels() + c];
                output.data[y * output.step + output.channels() * x + c] =
                        static_cast<uchar>(backgroundPx * (1. - opacity) + foregroundPx * opacity);
            }
        }
    }
}

void soundLocalizerModule::drawLeft(cv::Mat &img) {
    overlayImage(img, speakerImg, img, cv::Point(21, 7));
}

void soundLocalizerModule::drawRight(cv::Mat &img) {
    overlayImage(img, speakerImg, img, cv::Point(256, 7));
}

void soundLocalizerModule::writeImage() {

    if (outputImagePort.getOutputCount()) {

        if (drawOnLeft) {
            drawLeft(*template_img);
        } else if (drawOnRight) {
            drawRight(*template_img);
        }
        yarp::sig::ImageOf<yarp::sig::PixelRgb> &outputYarpImage = outputImagePort.prepare();
        outputYarpImage.resize(template_img->cols, template_img->rows);
        outputYarpImage = yarp::cv::fromCvMat<yarp::sig::PixelRgb>(*template_img);
        outputImagePort.write();
    }
}



bool soundLocalizerModule::sendEmotion(std::string iCubFacePart, std::string emotionCmd) {
    if (clientRPCEmotion.getOutputCount()) {
        yarp::os::Bottle cmd;
        cmd.addString("set");
        cmd.addString(iCubFacePart);

        cmd.addString(emotionCmd);

        yarp::os::Bottle response;
        clientRPCEmotion.write(cmd, response);
        yDebug("Send %s face emotion get response : %s", cmd.toString().c_str(), response.toString().c_str());

        return response.toString().find("[ok]") != std::string::npos;
    }

    return false;

}