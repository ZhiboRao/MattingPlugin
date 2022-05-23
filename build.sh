#!/bin/bash
build_folder="build"

echo "Begin to build..."
if [ ! -d ${build_folder} ];then
    mkdir ${build_folder}
fi

cd ${build_folder}
# cmake -DCMAKE_PREFIX_PATH=/home3/raozhibo/Documents/Programs/MattingPlugin/3rd/libtorch ..
cmake ..
make
echo "Finish!"