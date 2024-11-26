build_all: lanczos_serial lanczos_omp lanczos_pthreads lanczos_memo

lanczos_pthreads:
	gcc -o ../bin/lanczos_pthreads lanczos_pthreads.c -lm -lpng -lpthread -fopenmp

lanczos_cache:
	gcc -o ../bin/lanczos_cache lanczos_cache.c -lm -lpng

lanczos_memo:
	gcc -o ../bin/lanczos_memo lanczos_memo.c -lm -lpng

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

run_memo:
	../bin/lanczos_memeo

clean:
	rm -f ../bin/*
	rm -f *.txt

run_batch:
	python3 batch.py >> batch.txt

.PHONY: all clean
