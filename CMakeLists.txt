# Copyright: (C) 2020 RBCS Robotics Brain and Cognitive Science
# Authors: Jonas Gonzalez
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

SET(KEYWORD "soundLocalizer")
PROJECT(${KEYWORD})
cmake_minimum_required(VERSION 3.2)


set(ICUB_CONTRIB_DIRS $ENV{ICUB_DIR}/include)
find_package(YARP REQUIRED COMPONENTS sig cv)
FIND_PACKAGE(ICUBcontrib REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)


list(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${ICUB_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${ICUBCONTRIB_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(YarpInstallationHelpers)
include(ICUBcontribHelpers)
include(ICUBcontribOptions)

icubcontrib_set_default_prefix()
set(ICUB_APPLICATIONS_PREFIX "$ENV{ICUB_ROOT}" CACHE PATH "Application path prefix")


INCLUDE_DIRECTORIES(
        ${PROJECT_SOURCE_DIR}/include
        /usr/local/src/robot/yarp/src/libYARP_cv/src
        ${YARP_INCLUDE_DIRS}
        ${ICUB_INCLUDE_DIRS}
        ${ICUB_CONTRIB_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

# Search for source code.
FILE(GLOB folder_source src/*.cpp src/*.cc src/*.c)
FILE(GLOB folder_header include/iCub/*.h)
SOURCE_GROUP("Source Files" FILES ${folder_source})
SOURCE_GROUP("Header Files" FILES ${folder_header})

# Set up the main executable.
IF (folder_source)
    ADD_EXECUTABLE(${KEYWORD}
            ${folder_source}
            ${folder_header}
            src/soundLocalizer.cpp include/iCub/soundLocalizerModule.h)

    TARGET_LINK_LIBRARIES(${KEYWORD}
            ${OpenCV_LIBS}
            ${YARP_LIBRARIES}
            )

    INSTALL_TARGETS(/bin ${KEYWORD})
ELSE (folder_source)
    MESSAGE(FATAL_ERROR "No source code files found. Please add something")
ENDIF (folder_source)


add_subdirectory("app")
icubcontrib_add_uninstall_target()

#add_subdirectory(python-scripts)
