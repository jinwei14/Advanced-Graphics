# Makefile template for shared library

CXX = g++ # C++ compiler
CFLAGS = -I/usr/include/python2.7 -fPIC -O2 -g -std=c99# C flags
LDFLAGS = -shared -lpython2.7 # linking flags
RM = rm -f  # rm command
TARGET_LIB = libPNM.so # target lib
TARGET_MAIN = AG_CW1

SRCS = PNM.c # source files
OBJS = $(SRCS:.c=.o)

#MAIN_SRCS = main.cpp $(SRCS)
MAIN_OBJS = PNM.o main.o

.PHONY: all
all: $(TARGET_MAIN)

python: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	$(CXX) ${LDFLAGS} -o $@ $^

$(TARGET_MAIN): $(MAIN_OBJS)
	$(CXX) -o $@ PNM.c main.cpp

PNM.o: PNM.c
	$(CC) -c -o $@ $< $(CFLAGS)

main.o: main.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

$(SRCS:.c=.d):%.d:%.c
	$(CXX) $(CFLAGS) -MM $< >$@

include $(SRCS:.c=.d)

.PHONY: clean
clean:
	-${RM} ${TARGET_LIB} ${MAIN_OBJS} $(SRCS:.c=.d) $(TARGET_MAIN) *~ *.pyc
