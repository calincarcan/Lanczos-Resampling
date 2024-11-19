all: run

lanczos: lanczos.c
	gcc -o ../bin/lanczos lanczos.c -lm -lpng

run: lanczos
	../bin/lanczos 512 4096

clean:
	rm -f ../bin/*
	rm -f *.txt