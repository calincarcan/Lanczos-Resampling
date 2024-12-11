// Calin

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

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

#define LANCZOS_RADIUS 3
#define MASTER 0

double lanczos_kernel(double x) {
    int a = LANCZOS_RADIUS;
    if (x == 0) {
        return 1.0;
    } else if (x == a || x == -a) {
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
            result += weight * *(data + i * width + j);  // Access through pointer arithmetic
            weight_sum += weight;
        }
    }

    if (weight_sum != 0) {
        result /= weight_sum;
    }

    return (int)result;
}

void* lanczos_thread_function(void* args) {
    ThreadArgs* targs = (ThreadArgs*)args;

    for (int i = targs->start_row; i < targs->end_row; i++) {
        for (int j = 0; j < targs->new_width; j++) {
            double x = i * targs->scale_height;
            double y = j * targs->scale_width;
            int result = lanczos_2d_interpolate(targs->data, targs->height, targs->width, x, y);
            *(targs->output + (i - targs->start_row) * targs->new_width + j) = fmax(fmin(result, 255), 0); // Clamp values using pointer arithmetic
        }
    }

    pthread_exit(NULL);
}

void apply_2d_lanczos_mpi(int *data, int height, int width, int *output, int new_height, int new_width, int rank, int size) {
    double scale_height = (double)(height - 1) / (new_height - 1);
    double scale_width = (double)(width - 1) / (new_width - 1);

    int rows_per_process = new_height / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? new_height : start_row + rows_per_process;
    int local_rows = end_row - start_row;

    int thread_count = 4;
    pthread_t threads[thread_count];
    ThreadArgs thread_args[thread_count];

    int rows_per_thread = local_rows / thread_count;

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

        thread_args[t].start_row = start_row + t * rows_per_thread;
        thread_args[t].end_row = (t == thread_count - 1) ? end_row : thread_args[t].start_row + rows_per_thread;

        pthread_create(&threads[t], NULL, lanczos_thread_function, &thread_args[t]);
    }

    // Wait for threads to finish
    for (int t = 0; t < thread_count; t++) {
        pthread_join(threads[t], NULL);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <initial_size> <final_size>\n", argv[0]);
        return 1;
    }
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int height = atoi(argv[1]), width = atoi(argv[1]); // Input image dimensions
    int new_height = atoi(argv[2]), new_width = atoi(argv[2]); // Output image dimensions

    int *data = NULL;
    int *local_output;

    if (rank == MASTER) {
        // Master initializes the input image in one contiguous block
        data = malloc(height * width * sizeof(int));
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                *(data + i * width + j) = rand() % 256;
            }
        }
    }

    // Each process gets its portion of rows
    int rows_per_process = new_height / size;
    int local_rows = (rank == size - 1) ? (new_height - rank * rows_per_process) : rows_per_process;

    local_output = malloc(local_rows * new_width * sizeof(int));

    // Broadcast the input image dimensions
    MPI_Bcast(&height, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Broadcast the input image to all processes
    if (rank != MASTER) {
        data = malloc(height * width * sizeof(int));
    }
    MPI_Bcast(data, height * width, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Perform local computations
    apply_2d_lanczos_mpi(data, height, width, local_output, new_height, new_width, rank, size);

    // Gather results from all processes to the master
    int *final_output = NULL;
    if (rank == MASTER) {
        final_output = malloc(new_height * new_width * sizeof(int));
    }

    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == MASTER) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int local_rows_i = (i == size - 1) ? (new_height - i * rows_per_process) : rows_per_process;
            recvcounts[i] = local_rows_i * new_width;
            displs[i] = i * rows_per_process * new_width;
        }
    }

    MPI_Gatherv(local_output, local_rows * new_width, MPI_INT,
                final_output, recvcounts, displs, MPI_INT, MASTER, MPI_COMM_WORLD);

    free(local_output);
    if (rank == MASTER) {
        // printf("Original image:\n");
        // for (int i = 0; i < height; i++) {
        //     for (int j = 0; j < width; j++) {
        //         printf("%d ", data[i * width + j]);
        //     }
        //     printf("\n");
        // }


        // for (int i = 0; i < new_height; i++) {
        //     for (int j = 0; j < new_width; j++) {
        //         printf("%d ", final_output[i * new_width + j]);
        //     }
        //     printf("\n");
        // }

        free(data);
        free(final_output);
    }

    MPI_Finalize();
    return 0;
}
