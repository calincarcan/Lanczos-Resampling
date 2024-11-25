build_all: lanczos_serial lanczos_omp

lanczos_serial: lanczos_serial.c 
	gcc -o ../bin/lanczos_serial lanczos_serial.c -lm -lpng

lanczos_omp: lanczos_omp.c 
	gcc -o ../bin/lanczos_omp lanczos_omp.c -lm -lpng -fopenmp

run_lanczos_serial: 
	../bin/lanczos_serial

run_omp: 
	../bin/lanczos_omp


clean:
	rm -f ../bin/*
	rm -f *.txt

.PHONY: all clean
