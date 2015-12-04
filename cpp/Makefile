CC = g++
CCFLAGS = -fPIC -O3 -Wall -ffast-math -msse -msse2 -fopenmp
LINKFLAGS = -shared -Wl -fopenmp -lgomp
INPUT = im2col.cpp
TARGET = libcpputil.so

# If we are going to use MKL, we include additional flags
MKL_FLAGS = -D DECAF_USE_MKL
MKL_LINK_FLAGS = -lmkl_rt

all: $(INPUT)
	$(CC) -c $(CCFLAGS) $(INPUT)
	$(CC) $(LINKFLAGS) -o $(TARGET) *.o

clean:
	rm *.so
	rm *.o
