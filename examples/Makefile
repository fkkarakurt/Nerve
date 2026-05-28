CC      := gcc
CFLAGS  := -O2 -Wall -W
LDFLAGS := -lm

.PHONY: all clean shared

all: xor sine iris regression dropout spiral model_io maintenance

xor:
	$(CC) $(CFLAGS) 01_xor.c                  -o xor         $(LDFLAGS)

sine:
	$(CC) $(CFLAGS) 02_sine.c                 -o sine        $(LDFLAGS)

iris:
	$(CC) $(CFLAGS) 03_iris.c                 -o iris        $(LDFLAGS)

regression:
	$(CC) $(CFLAGS) 05_regression.c           -o regression  $(LDFLAGS)

dropout:
	$(CC) $(CFLAGS) 06_dropout.c              -o dropout     $(LDFLAGS)

spiral:
	$(CC) $(CFLAGS) 07_spiral.c               -o spiral      $(LDFLAGS)

model_io:
	$(CC) $(CFLAGS) 08_model_io.c             -o model_io    $(LDFLAGS)

maintenance:
	$(CC) $(CFLAGS) 09_predictive_maintenance.c -o maintenance $(LDFLAGS)

# Shared library for the Python ctypes demo (Linux / macOS)
shared:
	$(CC) $(CFLAGS) -shared -fPIC -DNERVE_IMPLEMENTATION \
	    ../nerve.h -o libnerve.so $(LDFLAGS)

clean:
	-rm -f xor sine iris regression dropout spiral model_io maintenance \
	       libnerve.so nerve.dll *.net *.bin *.exe *.o
