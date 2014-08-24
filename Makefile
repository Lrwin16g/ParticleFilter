# Makefile

CC = g++
CFLAGS = -O
INCPATH += `pkg-config --cflags opencv`
LIBS += `pkg-config --libs opencv`

EXEDIR = ./bin
OBJDIR = ./obj
SRCDIR = ./src

TARGET1 = $(EXEDIR)/pftracking

OBJ1 = $(OBJDIR)/pftracking.o $(OBJDIR)/particlefilter.o $(OBJDIR)/statlib.o

all: $(TARGET1)

$(TARGET1): $(OBJ1)
	$(CC) $(LIBS) -o $(TARGET1) $^

$(OBJDIR)/pftracking.o: $(SRCDIR)/pftracking.cpp
	$(CC) $(CFLAGS) $(INCPATH) -c $< -o $@

$(OBJDIR)/particlefilter.o: $(SRCDIR)/particlefilter.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/statlib.o: $(SRCDIR)/statlib.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET1) $(OBJDIR)/*.o
