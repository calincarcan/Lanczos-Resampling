#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> 

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
 
int lanczos_2d_interpolate(int **data, int height, int width, double x, double y) {
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
            result += weight * data[i][j];
            weight_sum += weight;
        }
    }
 
    if (weight_sum != 0) {
        result /= weight_sum;
    }
 
    return (int)result;
}

void apply_2d_lanczos_mpi(int **data, int height, int width, int **output, int new_height, int new_width, int rank, int size) {
    double scale_height = (double)(height - 1) / (new_height - 1);
    double scale_width = (double)(width - 1) / (new_width - 1);

    int rows_per_process = new_height / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? new_height : start_row + rows_per_process;
    int local_rows = end_row - start_row;

    // OpenMP parallelization
    #pragma omp parallel for
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < new_width; j++) {
            double x = i * scale_height;
            double y = j * scale_width;
            int result = lanczos_2d_interpolate(data, height, width, x, y);
            output[i - start_row][j] = fmax(fmin(result, 255), 0); // Clamp values
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
 
    int height = 512, width = 512; // Input image dimensions
    int new_height = 4096, new_width = 4096; // Output image dimensions
 
    int **data = NULL; // Input image
    int **local_output; // Output image for this process
 
    if (rank == MASTER) {
        // Master initializes the input image
        data = malloc(height * sizeof(int *));
        for (int i = 0; i < height; i++) {
            data[i] = malloc(width * sizeof(int));
            for (int j = 0; j < width; j++) {
                data[i][j] = rand() % 256;
            }
        }
    }
 
    // Each process gets its portion of rows
    int rows_per_process = new_height / size;
    int local_rows = (rank == size - 1) ? (new_height - rank * rows_per_process) : rows_per_process;
 
    local_output = malloc(local_rows * sizeof(int *));
    for (int i = 0; i < local_rows; i++) {
        local_output[i] = malloc(new_width * sizeof(int));
    }
 
    // Broadcast the input image dimensions
    MPI_Bcast(&height, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
 
    // Broadcast the input image to all processes
    if (rank != MASTER) {
        data = malloc(height * sizeof(int *));
        for (int i = 0; i < height; i++) {
            data[i] = malloc(width * sizeof(int));
        }
    }
    for (int i = 0; i < height; i++) {
        MPI_Bcast(data[i], width, MPI_INT, MASTER, MPI_COMM_WORLD);
    }
 
    // Perform local computations
    apply_2d_lanczos_mpi(data, height, width, local_output, new_height, new_width, rank, size);
 
    // Gather results from all processes to the master
    int **final_output = NULL;
    if (rank == MASTER) {
        final_output = malloc(new_height * sizeof(int *));
        for (int i = 0; i < new_height; i++) {
            final_output[i] = malloc(new_width * sizeof(int));
        }
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
 
    int *local_output_flat = malloc(local_rows * new_width * sizeof(int));
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < new_width; j++) {
            local_output_flat[i * new_width + j] = local_output[i][j];
        }
    }
 
    int *final_output_flat = NULL;
    if (rank == MASTER) {
        final_output_flat = malloc(new_height * new_width * sizeof(int));
    }
 
    MPI_Gatherv(local_output_flat, local_rows * new_width, MPI_INT,
                final_output_flat, recvcounts, displs, MPI_INT, MASTER, MPI_COMM_WORLD);
 
    // Reconstruct 2D array in master
    if (rank == MASTER) {
        for (int i = 0; i < new_height; i++) {
            for (int j = 0; j < new_width; j++) {
                final_output[i][j] = final_output_flat[i * new_width + j];
            }
        }
        printf("Resampling completed.\n");
    }
 
    // Clean up
    for (int i = 0; i < local_rows; i++) {
        free(local_output[i]);
    }
    free(local_output);
 
    if (rank == MASTER) {
        for (int i = 0; i < height; i++) {
            free(data[i]);
        }
        free(data);
 
        for (int i = 0; i < new_height; i++) {
            free(final_output[i]);
        }
        free(final_output);
    }
 
    MPI_Finalize();
    return 0;
}