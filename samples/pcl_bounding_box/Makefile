TARGET= pcl_bounding_box
CXX= g++
CXXFLAGS= -O3 -Wall \
	  -std=c++11 \
	  -I/usr/local/include \
	  -I/usr/include/eigen3 \
	  -I/usr/include/vtk-5.8/ \
	  -I/usr/include/pcl-1.7
CXXLIBS= \
	 -L/usr/lib \
	 -L/usr/local/lib \
	 -L/usr/lib/x86_64-linux-gnu \
	 -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_core \
	 -lpcl_io -lpcl_io_ply -lpcl_common -lpcl_registration -lpcl_visualization -lpcl_kdtree \
	 -lboost_system


SRCS= $(wildcard ./*.cpp)
OBJS= $(patsubst %.cpp, %.o, $(notdir $(SRCS)))


$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(CXXLIBS)

./%.o: ./%.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

clean:
	rm -rf *.o $(TARGET)

.PHONY: clean

