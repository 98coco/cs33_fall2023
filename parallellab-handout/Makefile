CC = gcc
CFLAGS = -O1 -Wall -fopenmp
LIBS = -lm

ifdef GPROF
CFLAGS += -pg
endif

SRCS = main.c tests.c sequential.c parallel.c

test: $(SRCS) utils.h
	$(CC) $(CFLAGS) $(LIBS) -o test $(SRCS)

.PHONY: clean

clean:
	rm -f *.o test gmon.out *~
