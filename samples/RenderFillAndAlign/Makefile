TARGET= RenderFillAndAlign
CXX= g++
CXXFLAGS= -O3 -Wall \
	  -std=c++11 \
	  -I/usr/local/include/
CXXLIBS= \
	 -L/usr/local/lib \
	 -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_core


SRCS= $(wildcard ./*.cpp)
OBJS= $(patsubst %.cpp, %.o, $(notdir $(SRCS)))


$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(CXXLIBS)

./%.o: ./%.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)


clean:
	rm -rf *.o $(TARGET)

.PHONY: clean

