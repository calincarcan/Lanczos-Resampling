build_all: lanczos_serial lanczos_omp lanczos_pthreads lanczos_memo

lanczos_pthreads:
	gcc -o ../bin/lanczos_pthreads lanczos_pthreads.c -lm -lpng -lpthread -fopenmp

lanczos_cache:
	gcc -o ../bin/lanczos_cache lanczos_cache.c -lm -lpng

lanczos_memo:
	gcc -o ../bin/lanczos_memo lanczos_memo.c -lm -lpng


lanczos_serial: lanczos_serial.c 
	gcc -o ../bin/lanczos_serial lanczos_serial.c -lm

lanczos_omp: lanczos_omp.c 
	gcc -o ../bin/lanczos_omp lanczos_omp.c -lm -fopenmp
  
clean:
	rm -f ../bin/*
	rm -f *.txt *.out

run_batch:
	python3 batch.py >> batch.txt

.PHONY: all clean
