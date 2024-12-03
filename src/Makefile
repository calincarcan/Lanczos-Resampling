build_all: lanczos_serial lanczos_omp lanczos_mpi_threads

lanczos_serial: lanczos_serial.c 
	gcc -o ../bin/lanczos_serial lanczos_serial.c -lm

lanczos_omp: lanczos_omp.c 
	gcc -o ../bin/lanczos_omp lanczos_omp.c -lm -fopenmp

lanczos_mpi_threads: lanczos_mpi_threads.c
	mpicc -o ../bin/lanczos_mpi_threads lanczos_mpi_threads.c -lm

run_mpi_threads:
	mpirun -np 8 ../bin/lanczos_mpi_threads 100 1000

run_serial:
	../bin/lanczos_serial 100 200

run_omp:
	../bin/lanczos_omp 100 200

clean:
	rm -f ../bin/*
	rm -f *.txt *.out

.PHONY: all clean
