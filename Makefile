INCLUDE = $(shell pkg-config --cflags opencv)
LIBS = $(shell pkg-config --libs opencv)
OBJECTS = EulerianMotionMag.o vibe.o main.o
SOURCE = EulerianMotionMag.cpp vibe.cpp main.cpp

BIN = bin
$(OBJECTS) : $(SOURCE)        
	g++ -c $(SOURCE)
$(BIN):$(objects)
	g++ -o $(BIN) $(OBJECTS) -I $(INCLUDE) $(LIBS)

clean: 
	rm $(OBJECTS) $(BIN)                                 
