# Copyright: (C) 2020 RBCS Robotics Brain and Cognitive Science
# Authors: Jonas Gonzalez
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

set(appname ${KEYWORD})

file(GLOB ${conf} DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${appname})
file(GLOB ${scripts} DESTINATION ${ICUBCONTRIB_APPLICATIONS_TEMPLATES_INSTALL_DIR})