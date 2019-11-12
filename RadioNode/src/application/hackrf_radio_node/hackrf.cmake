cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set Target Name
set(${TARGET_ID} hackrf_node)

# Include Source Directories
include_directories(
        ${CMAKE_SOURCE_DIR}/src/application/
        ${CMAKE_SOURCE_DIR}/src/libraries/
        ${CMAKE_SOURCE_DIR}/src/test/
        ${CMAKE_SOURCE_DIR}/third-party/
        ${CMAKE_SOURCE_DIR}/third-party/hackrf/
        ${CMAKE_SOURCE_DIR}/third-party/hackrf/hackrf/
        ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/
        ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/include/
        ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/convenience/
        ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/getopt/
        /opt/local/include/libhackrf/
        /usr/local/include/libusb-1.0/
        /usr/local/include)

# Includes
include(gatherSource)
include(apple_boost)


gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/common SRC_COMMON)
gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/DigitalSignalProcessing SRC_DSP)
gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/hackrf SRC_HACKRF)
gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/network SRC_NETWORK)
gatherThirdPartySource(${CMAKE_SOURCE_DIR}/third-party/FIR-filter-class SRC_TP_FIR_FILTER)
gatherThirdPartySource(${CMAKE_SOURCE_DIR}/third-party/hackrf/host/libhackrf/src SRC_TP_HACKRF)



message(">> >> >> ${SRC}")



## Add Unit Tests
add_executable(hackrf_radio_node ${CMAKE_SOURCE_DIR}/src/application/hackrf_radio_node/hackrf_main.cpp
               ${SRC_COMMON}
               ${SRC_DSP}
               ${SRC_HACKRF}
               ${SRC_NETWORK}
               ${SRC_TP_FIR_FILTER}
               ${SRC_TP_HACKRF})

target_link_libraries(hackrf_radio_node
        Boost::program_options
        Boost::system
        /usr/local/lib/libusb-1.0.dylib
        /usr/local/lib/libliquid.dylib)

