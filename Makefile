# gcc rather than cc: it resolves on Linux, on macOS (aliased to clang) and on
# a MinGW/MSYS2 Windows box, where cc does not exist. Note `:=`, not `?=` —
# make predefines CC as cc, so `?=` would never take effect. Override on the
# command line: `make CC=clang test`.
CC     := gcc
CFLAGS := -O2 -std=c99 -Wall -Wextra -pedantic

all:
	$(MAKE) -C examples

games:
	$(MAKE) -C examples games

test:
	$(CC) $(CFLAGS) tests/test_nerve.c -o tests/test_nerve -lm
	./tests/test_nerve

# The ANSI C89 claim in the README, enforced.
check-c89:
	$(CC) -O2 -std=c89 -pedantic-errors -Wall -Wextra -Werror \
	    examples/01_xor.c -o tests/nerve_c89 -lm
	@echo "nerve.h compiles as strict ANSI C89"

clean:
	$(MAKE) -C examples clean
	rm -f tests/test_nerve tests/test_nerve.exe tests/nerve_c89 tests/nerve_c89.exe

.PHONY: all games test check-c89 clean
