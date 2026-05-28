all:
	$(MAKE) -C examples

games:
	$(MAKE) -C examples games

clean:
	$(MAKE) -C examples clean

.PHONY: all games clean
