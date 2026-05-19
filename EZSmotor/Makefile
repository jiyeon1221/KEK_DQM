CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c99 $(shell pkg-config --cflags libmodbus)
LDFLAGS = $(shell pkg-config --libs libmodbus)

TARGET  = azd_kren
SRCS    = main.c azd_kren.c
OBJS    = $(SRCS:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.c azd_kren.h registers.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
