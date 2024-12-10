// Bogdan

#include <pthread.h>
#include <stdio.h>
#include <math.h>
#include <png.h>
#include <stdlib.h>
#include <omp.h>

#define LANCZOS_RADIUS 3
#define NUM_THREADS 8

// Memorizarea de valori pentru kernelul Lanczos
double lanczos_values[LANCZOS_RADIUS * 2 + 1][LANCZOS_RADIUS * 2 + 1];

double lanczos_kernel(double x) {
    if (lanczos_values[LANCZOS_RADIUS + (int)x][LANCZOS_RADIUS + (int)x] != 0) {
        return lanczos_values[LANCZOS_RADIUS + (int)x][LANCZOS_RADIUS + (int)x];
    }

    int a = LANCZOS_RADIUS;
    if (x == 0) {
        return 1.0; // sinc(0) = 1
    }
    else if (x == a || x == -a) {
        return 0.0; // Lanczos function for values of x = ±a
    }
    else {
        lanczos_values[LANCZOS_RADIUS + (int)x][LANCZOS_RADIUS + (int)x] = (sin(M_PI * x) / (M_PI * x)) * (sin(M_PI * x / a) / (M_PI * x / a));
        return (sin(M_PI * x) / (M_PI * x)) * (sin(M_PI * x / a) / (M_PI * x / a));
    }
}

typedef struct {
    int **data;
    int height;
    int width;
    int **output;
    int new_height;
    int new_width;
    int start_row;
    int end_row;
} ThreadData;

#pragma region NO_PARRALLEL

void print(int **a, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
}

#pragma endregion

int lanczos_2d_interpolate(int **data, int height, int width, double x, double y) {
    int a = LANCZOS_RADIUS;

    double result = 0;
    double weight_sum = 0;

    int center_x = (int)x;
    int center_y = (int)y;

    for (int i = center_x - a + 1; i < center_x + a; i++) {
        for (int j = center_y - a + 1; j < center_y + a; j++) {
            if (i < 0 || i >= height || j < 0 || j >= width)
                continue;
            double weight = lanczos_kernel(x - i) * lanczos_kernel(y - j);
            result += weight * data[i][j];
            weight_sum += weight;
        }
    }

    if (weight_sum != 0) {
        result /= weight_sum;
    }

    return result;
}

// Thread function to process a subset of rows
void *thread_func(void *arg) {
    ThreadData *thread_data = (ThreadData *)arg;

    int **data = thread_data->data;
    int height = thread_data->height;
    int width = thread_data->width;
    int **output = thread_data->output;
    int new_height = thread_data->new_height;
    int new_width = thread_data->new_width;
    int start_row = thread_data->start_row;
    int end_row = thread_data->end_row;

    double scale_height = (double)(height - 1) / (new_height - 1);
    double scale_width = (double)(width - 1) / (new_width - 1);

#pragma omp parallel for collapse(2) schedule(dynamic) // Parallelize the nested loops
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < new_width; j++) {
            double x = i * scale_height;
            double y = j * scale_width;
            int result = lanczos_2d_interpolate(data, height, width, x, y);
            result = fmax(fmin(result, 255), 0);
            output[i][j] = result;
        }
    }

    return NULL;
}

void apply_2d_lanczos(int **data, int height, int width, int **output, int new_height, int new_width) {
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    int rows_per_thread = new_height / NUM_THREADS;
    int extra_rows = new_height % NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        thread_data[t].data = data;
        thread_data[t].height = height;
        thread_data[t].width = width;
        thread_data[t].output = output;
        thread_data[t].new_height = new_height;
        thread_data[t].new_width = new_width;
        thread_data[t].start_row = t * rows_per_thread;
        thread_data[t].end_row = (t + 1) * rows_per_thread;

        if (t == NUM_THREADS - 1) {
            thread_data[t].end_row += extra_rows; // Handle remaining rows in the last thread
        }

        pthread_create(&threads[t], NULL, thread_func, &thread_data[t]);
    }

    // Wait for all threads to finish
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
}

int main(int argc, char *argv[]) {
    int width, height;
    width = atoi(argv[1]);
    height = atoi(argv[1]);
    int **image = malloc(height * sizeof(int *));
    for (int i = 0; i < height; i++) {
        image[i] = malloc(width * sizeof(int));
        for (int j = 0; j < width; j++) {
            image[i][j] = rand() % 256;
        }
    }

    int new_width = atoi(argv[2]);
    int new_height = atoi(argv[2]);

    int **new_image = calloc(new_height, sizeof(int *));
    for (int i = 0; i < new_height; i++)
        new_image[i] = calloc(new_width, sizeof(int));

    apply_2d_lanczos(image, height, width, new_image, new_height, new_width);

    // Free the memory allocated for the images

    for (int i = 0; i < height; i++) {
        free(image[i]);
    }

    free(image);

    for (int i = 0; i < new_height; i++) {
        free(new_image[i]);
    }

    free(new_image);
    return 0;
}
