CC := gcc
CFLAGS += -O2 -g -Wall -W -I../src -Iocrlib
LIBS := -lm ../src/libnerveNET.a ocrlib/lib.a -lz

EXECUTABLES := ocr_train ocr_validate
 
all : $(EXECUTABLES)

ocr_train : ocr_train.o
	gcc -o $@ $^ $(LIBS)

ocr_validate : ocr_validate.o
	gcc -o $@ $^ $(LIBS)

network1.net : ocr_train
	./ocr_train -c -n32 -e0.025 $@ ../ocrdata/optdigits.training.gz \
	            ../ocrdata/optdigits.validation.gz

network2.net : ocr_train
	../tests/create_network -o $@ 64 16 16 10
	./ocr_train -e0.025 $@ ../ocrdata/optdigits.training.gz \
	            ../ocrdata/optdigits.validation.gz

indent :
	indent -TFILE -TgzFile -Ttraining_data_t --no-tabs *.[ch]

clean : 
	-rm -f $(EXECUTABLES)
	-rm -f *.o
	-rm -f network1.net network2.net

clean-backup :
	-rm -f *.bak *~

clean-all : clean clean-backup

source-dist : network1.net network2.net
	-rm -f $(EXECUTABLES)
	-rm -f *.o
	-rm -f *.bak *~
	-rm -f .cvsignore
