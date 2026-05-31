CC      := gcc
CFLAGS  := -O2 -Wall -W
LDFLAGS := -lm

.PHONY: all run clean

all: slope_stability slope_sensitivity

slope_stability:
	$(CC) $(CFLAGS) slope_stability.c -o slope_stability $(LDFLAGS)

slope_sensitivity:
	$(CC) $(CFLAGS) slope_sensitivity.c -o slope_sensitivity $(LDFLAGS)

# Build and run both studies (run from this folder so dataset/ resolves)
run: all
	./slope_stability
	./slope_sensitivity

clean:
	-rm -f slope_stability slope_sensitivity *.exe *.o
