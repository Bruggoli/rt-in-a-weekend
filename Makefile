CC = gcc
CFLAGS = -Wall -O3 -fopenmp
LDFLAGS = -lm -fopenmp
TARGET = main
SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) image.ppm

run: $(TARGET)
	./$(TARGET) > image.ppm

.PHONY: all clean run
