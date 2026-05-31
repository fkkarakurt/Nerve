CC      := gcc
CFLAGS  := -O2 -Wall -W
LDFLAGS := -lm

.PHONY: all data run clean

all: mnist

mnist:
	$(CC) $(CFLAGS) mnist.c -o mnist $(LDFLAGS)

# Fetch the dataset (needs internet; ~11 MB)
data:
	sh download_data.sh

# Build, ensure data is present, then train + evaluate
run: mnist
	@test -f data/train-images-idx3-ubyte || sh download_data.sh
	./mnist

clean:
	-rm -f mnist mnist.exe run_log.txt *.o
