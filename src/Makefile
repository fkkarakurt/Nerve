CC := gcc
CFLAGS += -g -O2 -Wall -W -Winline -finline-limit=1200

all : libnerveNET.a

libnerveNET.a : network.o
	ar crU $@ $^
	ranlib $@

interpolation.c : generate_sigma
	./generate_sigma > $@

network.o : interpolation.c

generate_sigma : generate_sigma.o
	$(CC) -o $@ $^ -lm

depend :
	makedepend *.c

indent :
	indent -Tneuron_t -Tlayer_t -Tnetwork_t -TFILE --no-tabs *.c

version:
	cp nerveNet.h nerveNet.h.bak
	cp network.c network.c.bak
	(VERSION=`cat ../VERSION` ; \
	sed "s/, version [0-9.]*/, version $$VERSION/;s/[0-9]* [A-Z][a-z]* 20[0-9]*/$$DATE/" nerveNet.h.bak > nerveNet.h ; \
	sed "s/, version [0-9.]*/, version $$VERSION/;s/[0-9]* [A-Z][a-z]* 20[0-9]*/$$DATE/" network.c.bak > network.c)

clean :
	-rm -f libnerveNET.a
	-rm -f interpolation.c 
	-rm -f network.o generate_sigma.o
	-rm -f generate_sigma

clean-backup :
	-rm -f *.bak *~

clean-depend :
	mv -f Makefile Makefile.bak
	awk '{print}/^# DO NOT DELETE THIS LINE/{exit}' Makefile.bak > Makefile
	-rm -f Makefile.bak

clean-all : clean clean-backup clean-depend

source-dist : interpolation.c
	-rm -f libnerveNET.a
	-rm -f network.o generate_sigma.o
	-rm -f generate_sigma
	-rm -f *.bak *~
	-rm -f .cvsignore

# DO NOT DELETE THIS LINE, MAKEDEPEND DEPENDS ON IT.
