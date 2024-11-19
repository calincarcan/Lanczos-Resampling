all: run

lanczos: lanczos.c
	gcc -o ../bin/lanczos lanczos.c -lm -lpng

run: lanczos
	../bin/lanczos

clean:
	rm -f ../bin/lanczos
	rm -f *.txt

.PHONY: all clean
