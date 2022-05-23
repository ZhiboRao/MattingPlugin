# show the message
message("MattingPlugin---${PROJECT_SOURCE_DIR}")

# set the output folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib) 

# the header and source files in this project
include_directories(./include)
aux_source_directory(./src src_dirs)

# torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# generate executeable file
add_library(MattingPlugin SHARED ${src_dirs})
target_link_libraries(MattingPlugin ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET MattingPlugin PROPERTY CXX_STANDARD 14)