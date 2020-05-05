#include <iCub/soundLocalizerModule.h>
#include <yarp/os/Log.h>

using namespace yarp::os;

int main(int argc, char *argv[]) {

    Network yarp;
    soundLocalizerModule module;

    ResourceFinder rf;
    rf.setVerbose(true);
    rf.setDefaultConfigFile("sound_localizer.ini");      //overridden by --from parameter
    rf.setDefaultContext("soundLocalizer");              //overridden by --context parameter
    rf.configure(argc, argv);

    yInfo("resourceFinder: %s", rf.toString().c_str());

    module.runModule(rf);
    return 0;
}
