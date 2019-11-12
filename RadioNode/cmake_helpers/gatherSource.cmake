# standard/ProcessorCount.cmake

function(gatherLibrarySource directory SOURCE)
    get_filename_component(TARGET_ID ${directory} NAME)
    string(REPLACE " " "_" TARGET_ID ${TARGET_ID})

    MESSAGE("Gathering source from ${TARGET_ID}")
    file(GLOB SRC_FILES ${directory}/*.cpp ${directory}/*.c ${directory}/*.hpp ${directory}/*.h)
    set("${SOURCE}" ${SRC_FILES} PARENT_SCOPE)
endfunction()


function(gatherThirdPartySource directory SOURCE)
    message("Gathering source from ${directory}")

    file(GLOB SRC_FILES
            ${directory}/*.cpp
            ${directory}/*.c
            ${directory}/*.hpp
            ${directory}/*.h)
    set("${SOURCE}" ${SRC_FILES} PARENT_SCOPE)
endfunction()