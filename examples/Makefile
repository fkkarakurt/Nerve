CC      := gcc
CFLAGS  := -O2 -Wall -W
LDFLAGS := -lm

.PHONY: all games clean

all: xor sine iris regression dropout spiral model_io maintenance

games: snake_ai pong_ai flappy_ai

xor:
	$(CC) $(CFLAGS) 01_xor.c                    -o xor          $(LDFLAGS)

sine:
	$(CC) $(CFLAGS) 02_sine.c                   -o sine         $(LDFLAGS)

iris:
	$(CC) $(CFLAGS) 03_iris.c                   -o iris         $(LDFLAGS)

regression:
	$(CC) $(CFLAGS) 05_regression.c             -o regression   $(LDFLAGS)

dropout:
	$(CC) $(CFLAGS) 06_dropout.c                -o dropout      $(LDFLAGS)

spiral:
	$(CC) $(CFLAGS) 07_spiral.c                 -o spiral       $(LDFLAGS)

model_io:
	$(CC) $(CFLAGS) 08_model_io.c               -o model_io     $(LDFLAGS)

maintenance:
	$(CC) $(CFLAGS) 09_predictive_maintenance.c -o maintenance  $(LDFLAGS)

snake_ai:
	$(CC) $(CFLAGS) 10_snake_ai.c               -o snake_ai     $(LDFLAGS)

pong_ai:
	$(CC) $(CFLAGS) 11_pong_ai.c                -o pong_ai      $(LDFLAGS)

flappy_ai:
	$(CC) $(CFLAGS) 12_flappy_ai.c              -o flappy_ai    $(LDFLAGS)

clean:
	-rm -f xor sine iris regression dropout spiral model_io maintenance \
	       snake_ai pong_ai flappy_ai \
	       *.net *.bin *.exe *.o
