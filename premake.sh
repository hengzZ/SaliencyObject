#!/bin/bash

mkdir -p build
cd build 

cmake ../3rdparty/gSLICr
make

cd ..
