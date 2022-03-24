EXECUTABLES := create_network copy_network show_network train_network \
	jolt_network
CC := gcc
CFLAGS += -O2 -g -Wall -W -I../src
LIBS := -lm ../src/nerveNet.a

all : $(EXECUTABLES) xornet.net

create_network : create_network.o
	$(CC) -o $@ $^ $(LIBS)

copy_network : copy_network.o
	$(CC) -o $@ $^ $(LIBS)

show_network : show_network.o spec.o
	$(CC) -o $@ $^ $(LIBS)

train_network : train_network.o spec.o
	$(CC) -o $@ $^ $(LIBS) 

jolt_network : jolt_network.o
	$(CC) -o $@ $^ $(LIBS) 

xornet.net : train_network xornet.spec
	./train_network -n3 -e0.0001 xornet.spec -oxornet.net > /dev/null

indent :
	indent -TFILE --no-tabs *.[ch]

clean : 
	-rm -f *.o $(EXECUTABLES)
	-rm -f xornet.net

clean-backup :
	-rm -f *.bak *~

clean-all : clean clean-backup

source-dist : xornet.net
	-rm -f *.o $(EXECUTABLES)
	-rm -f *.bak *~
	-rm -f .cvsignore
	
