cmake_minimum_required(VERSION 2.8.6)
project(ilastikbackend)

# ------------------------------------------
# user defined configuration
set(WITH_TESTS "false" CACHE BOOL "Build tests.")
set(WITH_PYTHON "true" CACHE BOOL "Build python wrapper.")
set(WITH_COMMAND_LINE_TOOLS "false" CACHE BOOL "")

# ------------------------------------------
# global configuration
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-local-typedefs")

# check for C++ 11 support:
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++1z" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX17)
    add_definitions(-DCPP17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
elseif(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(SEND_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# we use optional, etc., and hence need to decide between std and boost
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  add_definitions(-DCPP17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# C++ STL debug symbols
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(WITH_DEBUG_STL "False" CACHE BOOL "Build with C++ stl debug symbols?")
  if(WITH_DEBUG_STL)
    add_definitions(-D_GLIBCXX_DEBUG)
  endif()
endif()

# ------------------------------------------
# find dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_extensions/)
find_package(Boost REQUIRED filesystem)
find_package(VIGRA)
find_package(HDF5 REQUIRED COMPONENTS HL)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${VIGRA_INCLUDE_DIR}
)
# ------------------------------------------
# build library (only really contains the code for loading random forests...
file(GLOB_RECURSE SOURCES src/*.cpp)
file(GLOB_RECURSE HEADERS include/*.h)
add_library(ilastikbackend SHARED ${SOURCES} ${HEADERS})
target_link_libraries(ilastikbackend ${Boost_LIBRARIES} ${VIGRA_IMPEX_LIBRARY} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

# installation
install(TARGETS ilastikbackend 
		ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" 
		DESTINATION include/ilastikbackend 
		PATTERN ".h" 
		PATTERN ".git" EXCLUDE)

# ------------------------------------------
# add subdirectories
if(WITH_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()

if(WITH_PYTHON)
    add_subdirectory(externals/pybind11)
    add_subdirectory(python)
endif()

if(WITH_COMMAND_LINE_TOOLS)
    add_subdirectory(bin)
endif()
