# The Feynman benchmark.
#
#   make            build
#   make quick      12-equation smoke test (~10 s)
#   make full       all 100 equations, results.csv alongside (minutes)
#   make front EQ=I.6.2a    one equation, with its whole Pareto front

# gcc rather than cc, for the same reason as the root Makefile: it resolves on
# Linux, on macOS (aliased to clang) and on a MinGW/MSYS2 Windows box, where cc
# does not exist. `:=`, not `?=`, because make predefines CC as cc. Override on
# the command line: `make CC=clang full`.
CC     := gcc
CFLAGS := -O2 -std=c99 -Wall -Wextra -pedantic
LDLIBS  = -lm
EQ     ?= I.6.2a

feynman: feynman.c equations.c equations.h ../../nerve_discover.h
	$(CC) $(CFLAGS) equations.c feynman.c -o $@ $(LDLIBS)

quick: feynman
	./feynman --quick

full: feynman
	./feynman --csv results.csv

front: feynman
	./feynman --only $(EQ) --front

clean:
	rm -f feynman feynman.exe results.csv

.PHONY: quick full front clean
