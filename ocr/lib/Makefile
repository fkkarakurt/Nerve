CC := gcc
CFLAGS += -O2 -g -Wall -W -I../../src

lib.a : train.o sample.o fileio.o
	ar cru $@ $^
	ranlib $@
 
indent :
	indent -TFILE -TgzFile -Ttraining_data_t -Tnetwork_t --no-tabs *.[ch]

clean : 
	-rm -f *.o lib.a

clean-backup :
	-rm -f *.bak *~

clean-all : clean clean-backup

source-dist :
	-rm -f *.o lib.a
	-rm -f *.bak *~
	-rm -f .cvsignore
