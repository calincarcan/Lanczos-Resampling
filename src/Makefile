all: lanczos

lanczos: lanczos.c
	gcc -o ../bin/lanczos lanczos.c -lm -lpng

run: lanczos
	../bin/lanczos

clean:
	rm -f ../bin/lanczos
	rm -f *.txt

.PHONY: all clean

lanczos_threads: lanczos_threads.c
	gcc -o ../bin/lanczos_threads lanczos_threads.c -lm -lpng -pthread

run_thread: lanczos_threads
	../bin/lanczos_threads

clean_thread:
	rm -f ../bin/lanczos_threads
	rm -f *.txt