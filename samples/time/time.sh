#!/bin/bash

~/project/caffe/build/tools/caffe time --model=$1 -gpu 0 -iterations 100
