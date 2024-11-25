all: run

lanczos: lanczos.c
	gcc -o ../bin/lanczos lanczos.c -lm -lpng

run: lanczos
	../bin/lanczos 100 200

clean:
	rm -f ../bin/*
	rm -f *.txt

