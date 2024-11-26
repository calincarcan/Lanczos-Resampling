build_all: lanczos_serial lanczos_omp lanczos_pthreads lanczos_cache

lanczos_pthreads: lanczos_pthreads.c 
	gcc -o ../bin/lanczos_pthreads lanczos_pthreads.c -lm -lpng -lpthread -fopenmp

lanczos_cache:
	gcc -o ../bin/lanczos_cache lanczos_cache.c -lm -lpng


lanczos_serial: lanczos_serial.c 
	gcc -o ../bin/lanczos_serial lanczos_serial.c -lm -lpng

lanczos_omp: lanczos_omp.c 
	gcc -o ../bin/lanczos_omp lanczos_omp.c -lm -lpng -fopenmp

run_lanczos_cache:
	../bin/lanczos_cache

run_lanczos_serial: 
	../bin/lanczos_serial

run_omp: 
	../bin/lanczos_omp

run_pthreads: 
	../bin/lanczos_pthreads

clean:
	rm -f ../bin/*
	rm -f *.txt

.PHONY: all clean
