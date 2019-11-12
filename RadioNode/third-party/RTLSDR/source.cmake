cmake_minimum_required(VERSION 3.14)

# Set Target Name
get_filename_component(TARGET_ID ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" TARGET_ID ${TARGET_ID})

# Includes
include(${CMAKE_SOURCE_DIR}/cmake_helpers/directories)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Generate Target Source
file(GLOB ${TARGET_ID}_source *.cpp *.c)

