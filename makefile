CC=/opt/homebrew/bin/gcc-15
CFLAGS=-Wall -O3 

# Add library path for M1/M2 Macs
LDFLAGS=-L/opt/homebrew/opt/libomp/lib
INCLUDES=-I/opt/homebrew/opt/libomp/include
deps = ./core/*.h ./accel/*.h ./utils/*.h ./textures/*.h ./materials/*.h ./geometries/*.h

OBJ = main.c ./core/*.c ./accel/*.c ./utils/*.c ./textures/*.c ./materials/*.c ./geometries/*.c

LIBS= -lm -fopenmp

IMG_OUT = image.ppm

TARGET = main

make: $(OBJ)
		$(CC) $(INCLUDES) $(LDFLAGS) $(LIBS) $(OBJ) -o $(TARGET) $(CFLAGS)
run: export RTW_IMAGES = ./resources/
run: $(TARGET)
	   ./$(TARGET) > $(IMG_OUT)

performance:$(TARGET)
	   time perf record ./$(TARGET) > $(IMG_OUT)

1c:$(TARGET)
	   time OMP_NUM_THREADS=1 perf record ./$(TARGET) > $(IMG_OUT)

12c:$(TARGET)
	   time OMP_NUM_THREADS=12 perf record  ./$(TARGET) > $(IMG_OUT)
