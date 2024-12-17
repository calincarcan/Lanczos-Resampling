build_all: lanczos_memo lanczos_mpi_omp lanczos_mpi_pthreads lanczos_mpi_threads lanczos_omp lanczos_pthreads lanczos_serial lanczos_mpi_thr_pts lanczos_pthreads_omp lanczos_phtreads_omp_ptr

lanczos_phtreads_omp_ptr:
	gcc -o ../bin/lanczos_phtreads_omp_ptr lanczos_pthreads_omp_ptr.c -lm -lpthread -fopenmp

lanczos_pthreads_omp:
	gcc -o ../bin/lanczos_pthreads_omp lanczos_pthreads_omp.c -lm -lpthread -fopenmp

lanczos_mpi_thr_pts: lanczos_mpi_thr_pts.c
	mpicc -o ../bin/lanczos_mpi_thr_pts lanczos_mpi_thr_pts.c -lm

lanczos_memo:
	gcc -o ../bin/lanczos_memo lanczos_memo.c -lm

lanczos_mpi_omp:
	mpicc -o ../bin/lanczos_mpi_omp lanczos_mpi_omp.c -lm -fopenmp

lanczos_mpi_pthreads:
	mpicc -o ../bin/lanczos_mpi_pthreads lanczos_mpi_pthreads.c -lm -lpthread

lanczos_mpi_threads:
	mpicc -o ../bin/lanczos_mpi_threads lanczos_mpi_threads.c -lm -lpthread

lanczos_omp:
	gcc -o ../bin/lanczos_omp lanczos_omp.c -lm -fopenmp

lanczos_pthreads:
	gcc -o ../bin/lanczos_pthreads lanczos_pthreads.c -lm -lpthread

lanczos_serial:
	gcc -o ../bin/lanczos_serial lanczos_serial.c -lm
  
clean:
	rm -f ../bin/*

run_batch:
	python3 batch.py >> batch.txt

.PHONY: all clean
