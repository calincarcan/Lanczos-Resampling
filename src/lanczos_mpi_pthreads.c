// Adelina

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#define LANCZOS_RADIUS 3
#define MASTER 0

typedef struct {
    int thread_id;
    int thread_count;
    int *data;
    int height;
    int width;
    int *output;
    int new_height;
    int new_width;
    int start_row;
    int end_row;
    double scale_height;
    double scale_width;
} ThreadArgs;

double lanczos_kernel(double x) {
    int a = LANCZOS_RADIUS;
    if (x == 0) {
        return 1.0;
    } else if (x >= a || x <= -a) {
        return 0.0;
    } else {
        return (sin(M_PI * x) / (M_PI * x)) * (sin(M_PI * x / a) / (M_PI * x / a));
    }
}

int lanczos_2d_interpolate(int *data, int height, int width, double x, double y) {
    int a = LANCZOS_RADIUS;
    double result = 0;
    double weight_sum = 0;
    int center_x = (int)x;
    int center_y = (int)y;

    for (int i = center_x - a + 1; i < center_x + a; i++) {
        for (int j = center_y - a + 1; j < center_y + a; j++) {
            if (i < 0 || i >= height || j < 0 || j >= width) {
                continue;
            }
            double weight = lanczos_kernel(x - i) * lanczos_kernel(y - j);
            result += weight * data[i * width + j];
            weight_sum += weight;
        }
    }

    if (weight_sum != 0) {
        result /= weight_sum;
    }

    return (int)fmax(fmin(result, 255), 0); // Clamp to [0, 255]
}

void *lanczos_thread_function(void *args) {
    ThreadArgs *targs = (ThreadArgs *)args;

    for (int i = targs->start_row; i < targs->end_row; i++) {
        for (int j = 0; j < targs->new_width; j++) {
            double x = i * targs->scale_height;
            double y = j * targs->scale_width;
            int result = lanczos_2d_interpolate(targs->data, targs->height, targs->width, x, y);
            targs->output[(i - targs->start_row) * targs->new_width + j] = result;
        }
    }

    pthread_exit(NULL);
}

void apply_2d_lanczos_mpi(int *data, int height, int width, int *output, int new_height, int new_width, int rank, int size) {
    double scale_height = (double)(height - 1) / (new_height - 1);
    double scale_width = (double)(width - 1) / (new_width - 1);

    int rows_per_process = new_height / size;
    int extra_rows = new_height % size;
    int start_row = rank * rows_per_process + (rank < extra_rows ? rank : extra_rows);
    int end_row = start_row + rows_per_process + (rank < extra_rows ? 1 : 0);
    int local_rows = end_row - start_row;

    int thread_count = 4;
    pthread_t threads[thread_count];
    ThreadArgs thread_args[thread_count];

    int rows_per_thread = local_rows / thread_count;
    int extra_rows_thread = local_rows % thread_count;

    for (int t = 0; t < thread_count; t++) {
        thread_args[t].thread_id = t;
        thread_args[t].thread_count = thread_count;
        thread_args[t].data = data;
        thread_args[t].height = height;
        thread_args[t].width = width;
        thread_args[t].output = output;
        thread_args[t].new_height = new_height;
        thread_args[t].new_width = new_width;
        thread_args[t].scale_height = scale_height;
        thread_args[t].scale_width = scale_width;

        thread_args[t].start_row = start_row + t * rows_per_thread + (t < extra_rows_thread ? t : extra_rows_thread);
        thread_args[t].end_row = thread_args[t].start_row + rows_per_thread + (t < extra_rows_thread ? 1 : 0);

        if (pthread_create(&threads[t], NULL, lanczos_thread_function, &thread_args[t]) != 0) {
            fprintf(stderr, "Error creating thread %d\n", t);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    for (int t = 0; t < thread_count; t++) {
        if (pthread_join(threads[t], NULL) != 0) {
            fprintf(stderr, "Error joining thread %d\n", t);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int height = 4096, width = 4096;           // Input image dimensions
    int new_height = 32768, new_width = 32768; // Output image dimensions

    int *data = NULL;  // Input image
    int *local_output; // Output image for this process

    int rows_per_process = new_height / size;
    int extra_rows = new_height % size;
    int start_row = rank * rows_per_process + (rank < extra_rows ? rank : extra_rows);
    int local_rows = rows_per_process + (rank < extra_rows ? 1 : 0);

    local_output = (int *)malloc(local_rows * new_width * sizeof(int));
    if (!local_output) {
        fprintf(stderr, "Error allocating local_output\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == MASTER) {
        data = (int *)malloc(height * width * sizeof(int));
        if (!data) {
            fprintf(stderr, "Error allocating data\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                data[i * width + j] = rand() % 256;
            }
        }
    } else {
        data = (int *)malloc(height * width * sizeof(int));
        if (!data) {
            fprintf(stderr, "Error allocating data on process %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Bcast(data, height * width, MPI_INT, MASTER, MPI_COMM_WORLD);

    apply_2d_lanczos_mpi(data, height, width, local_output, new_height, new_width, rank, size);

    int *final_output = NULL;
    if (rank == MASTER) {
        final_output = (int *)malloc(new_height * new_width * sizeof(int));
        if (!final_output) {
            fprintf(stderr, "Error allocating final_output\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    if (!recvcounts || !displs) {
        fprintf(stderr, "Error allocating recvcounts or displs\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        int start = i * rows_per_process + (i < extra_rows ? i : extra_rows);
        int rows = rows_per_process + (i < extra_rows ? 1 : 0);
        recvcounts[i] = rows * new_width;
        displs[i] = start * new_width;
    }

    MPI_Gatherv(local_output, local_rows * new_width, MPI_INT,
                final_output, recvcounts, displs, MPI_INT, MASTER, MPI_COMM_WORLD);

    free(local_output);
    free(data);
    free(recvcounts);
    free(displs);
    if (rank == MASTER) {
        free(final_output);
    }

    MPI_Finalize();
    return 0;
}
