# Show Details
DETAILS= 1

ICAFFE=-I../caffe/include/ -I../caffe/build/src/
LCAFFE=-L../caffe/build/lib/
IOPENCV=-I/usr/local/include
LOPENCV=-L/usr/local/lib
IDIR=-I./3rdparty/gSLICr/
LDIR=-L./build/

SDIR=./src
ODIR=./obj

TARGET=demo
CXX=g++
CXXFLAGS= -O3 -Wall \
	  -std=c++11 \
	  $(IOPENCV) \
	  -I/usr/local/cuda/include \
	  $(ICAFFE) \
	  $(IDIR) \
	  -DUSE_OPENCV=1 
CXXLIBS= \
	 $(LOPENCV) \
	 -L/usr/local/cuda/lib64/ \
	 -L/usr/lib/x86_64-linux-gnu/ \
	 $(LCAFFE) \
	 $(LDIR) \
	 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs\
	 -lgSLICr_lib \
	 -lboost_system -lboost_thread -lboost_filesystem \
	 -lcaffe \
	 -lcudart \
	 -lglog \
	 -lprotobuf \
	 -lpthread


SRCS=$(wildcard $(SDIR)/*.cpp)
OBJS=$(patsubst %.cpp, $(ODIR)/%.o, $(notdir $(SRCS)))
DEPS=


ifeq ($(DETAILS), 1)
CXXFLAGS += -DDETAIL_SHOW
endif


all: dirs $(TARGET)

dirs:
	mkdir -p $(ODIR)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(CXXLIBS)

$(ODIR)/%.o: $(SDIR)/%.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)


clean:
	rm -rf $(ODIR) $(TARGET)

.PHONY: clean

