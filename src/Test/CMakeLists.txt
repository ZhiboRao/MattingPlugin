# show the message
message("Test---${PROJECT_SOURCE_DIR}")

# set the output folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# the header and source files in this project
include_directories(./include)
aux_source_directory(./src src_dirs)

# my lib
include_directories(${PROJECT_SOURCE_DIR}/src/MattingPlugin/include) 
link_directories(${PROJECT_SOURCE_DIR}/lib)

# torch lib
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# opencv lib
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# generate executeable file
add_executable(Test ${src_dirs})
target_link_libraries(Test MattingPlugin ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET Test PROPERTY CXX_STANDARD 14)