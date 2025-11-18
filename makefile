CC=clang
CFLAGS=-Wall -O3

deps = ./core/*.h ./accel/*.h ./utils/*.h ./textures/*.h ./materials/*.h ./geometries/*.h

OBJ = main.c ./core/*.c ./accel/*.c ./utils/*.c ./textures/*.c ./materials/*.c ./geometries/*.c

LIBS= -lm

IMG_OUT = image.ppm

TARGET = main

make: $(OBJ)
			$(CC) $(LIBS) $(OBJ) -o $(TARGET) $(CFLAGS)

run: export RTW_IMAGES = ./resources/
run: $(TARGET)
	   ./$(TARGET) > $(IMG_OUT)

performance:$(TARGET)
	   time perf record ./$(TARGET) > $(IMG_OUT)
