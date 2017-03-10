find_package( PythonInterp REQUIRED )
pybind11_add_module(pyilastikbackend ilastikbackend.cpp pyblocking.h pypixelclassification.h vigraconverter.h)
target_link_libraries(pyilastikbackend ilastikbackend)

# this will set the install dir to $PYTHON_SITE_PACKAGES
# make use of ${PYTHON_EXECUTABLE} instead of /usr/bin/python
if(NOT DEFINED PYILASTIKBACKEND_INSTALL_DIR OR PYILASTIKBACKEND_INSTALL_DIR MATCHES "^$")
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                    "from distutils.sysconfig import *; print get_python_lib(1)"
                    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
    file(TO_CMAKE_PATH ${PYTHON_SITE_PACKAGES} PYILASTIKBACKEND_INSTALL_DIR)
endif()

set(PYILASTIKBACKEND_INSTALL_DIR ${PYILASTIKBACKEND_INSTALL_DIR} CACHE PATH "where to install the dpct Python package" FORCE)

# this is the install path relative to CMAKE_INSTALL_PREFIX,
# use this in INSTALL() commands to get packaging right
file(RELATIVE_PATH PYILASTIKBACKEND_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} ${PYILASTIKBACKEND_INSTALL_DIR})

install(TARGETS pyilastikbackend
    RUNTIME DESTINATION ${PYILASTIKBACKEND_INSTALL_DIR}
    LIBRARY DESTINATION ${PYILASTIKBACKEND_INSTALL_DIR})
