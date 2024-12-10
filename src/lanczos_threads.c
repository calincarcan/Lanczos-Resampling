#include <stdio.h>
#include <math.h>
#include <png.h>
#include <stdlib.h>
#include <pthread.h>
#include

#define LANCZOS_RADIUS 3

typedef struct {
    int thread_id;
    int thread_count;
    int **data;
    int height;
    int width;
    int **output;
    int new_height;
    int new_width;
} ThreadArgs;

void* lanczos_thread_function(void* args) 
{
    ThreadArgs* targs = (ThreadArgs*)args;
    int thread_id = targs->thread_id;
    int thread_count = targs->thread_count;
    int **data = targs->data;
    int height = targs->height;
    int width = targs->width;
    int **output = targs->output;
    int new_height = targs->new_height;
    int new_width = targs->new_width;

    double scale_height = (double)(height - 1) / (new_height - 1);
    double scale_width = (double)(width - 1) / (new_width - 1);

    for (int i = thread_id; i < new_height; i += thread_count) {
        for (int j = 0; j < new_width; j++) {
            double x = i * scale_height;
            double y = j * scale_width;
            int result = lanczos_2d_interpolate(data, height, width, x, y);
            result = fmax(fmin(result, 255), 0);
            output[i][j] = result;
        }
    }
    pthread_exit(NULL);
}

int lanczos_2d_interpolate(int **data, int height, int width, double x, double y);

void write_png(const char* filename, int** matrix, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("File opening failed");
        return;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        fprintf(stderr, "Failed to create png struct\n");
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        fprintf(stderr, "Failed to create info struct\n");
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        fprintf(stderr, "Error during png creation\n");
        return;
    }

    png_init_io(png, fp);

    // Set the PNG header info for an RGBA image
    png_set_IHDR(
        png, info, width, height,
        8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    // Write each row of the matrix to the PNG file as RGBA values
    for (int y = 0; y < height; y++) {
        png_bytep row = (png_bytep)malloc(width * 4 * sizeof(png_byte)); // 4 bytes per pixel for RGBA
        for (int x = 0; x < width; x++) {
            int grayscale = matrix[y][x];
            grayscale = grayscale > 255 ? 255 : (grayscale < 0 ? 0 : grayscale); // Clamp to [0, 255]

            row[x*4 + 0] = grayscale; // Red
            row[x*4 + 1] = grayscale; // Green
            row[x*4 + 2] = grayscale; // Blue
            row[x*4 + 3] = 255;       // Alpha (fully opaque)
        }
        png_write_row(png, row);
        free(row);
    }

    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

double lanczos_kernel(double x) {
    int a = LANCZOS_RADIUS;
    if (x == 0) {
        return 1.0; // sinc(0) = 1
    } else if (x == a || x == -a) {
        return 0.0; // Lanczos function for values of x = ±a
    } else {
        return (sin(M_PI * x) / (M_PI * x)) * (sin(M_PI * x / a) / (M_PI * x / a));
    }
}

void print(int **a, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", a[i][j]);
        }
        printf("\n");
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

    return result;
}

void apply_2d_lanczos(int **data, int height, int width, int **output, int new_height, int new_width) {
    int a = LANCZOS_RADIUS;
    double scale_height = (double)(height - 1) / (new_height - 1);
    double scale_width = (double)(width - 1) / (new_width - 1);

    for (int i = 0; i < new_height; i++) {
        for (int j = 0; j < new_width; j++) {
            double x = i * scale_height;
            double y = j * scale_width;
            int result = lanczos_2d_interpolate(data, height, width, x, y);
            result = fmax(fmin(result, 255), 0);
            output[i][j] = result;

        }
    }
}

void apply_2d_lanczos_parallel(int **data, int height, int width, int **output, int new_height, int new_width, int thread_count) {
    pthread_t threads[thread_count];
    ThreadArgs thread_args[thread_count];

    for (int t = 0; t < thread_count; t++) {
        thread_args[t].thread_id = t;
        thread_args[t].thread_count = thread_count;
        thread_args[t].data = data;
        thread_args[t].height = height;
        thread_args[t].width = width;
        thread_args[t].output = output;
        thread_args[t].new_height = new_height;
        thread_args[t].new_width = new_width;

        int rc = pthread_create(&threads[t], NULL, lanczos_thread_function, (void*)&thread_args[t]);
        if (rc) {
            fprintf(stderr, "Error: pthread_create failed with code %d\n", rc);
            exit(EXIT_FAILURE);
        }
    }

    // Așteaptă finalizarea tuturor thread-urilor
    for (int t = 0; t < thread_count; t++) {
        pthread_join(threads[t], NULL);
    }
}


// DEPRECATED
int lanczos_1d_interpolate(int* data, int length, double x) {
    int a = LANCZOS_RADIUS;
    int i;
    double result = 0.0;
    int center = (int)x; // the integer part of x
    double sum = 0.0; // normalization factor (sum of kernels)

    // Sum up the values weighted by the Lanczos kernel
    for (i = -a + 1; i < a; i++) {
        if(center + i < 0 || center + i >= length)
            continue;
        double weight = lanczos_kernel(x - (center + i));
        result += weight * data[center + i];
        sum += weight;
    }

    // Normalize the result by the sum of the weights
    if (sum != 0) {
        result /= sum;
    }

    return result;
}

// DEPRECATED
void apply_1d_lanczos(int* data, int length, int* output, int output_length) {
    int a = LANCZOS_RADIUS;
    double scale = (double)(length - 1) / (output_length - 1); // scaling factor for resampling
    for (int i = 0; i < output_length; i++) {
        double x = i * scale;
        output[i] = lanczos_1d_interpolate(data, length, x);
    }
}

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = 512, height = 512;

    int new_width = 4096, new_height = 4096;

    int **image = NULL;
    int **new_image = NULL;

    if (rank == 0) {
        image = malloc(height * sizeof(int*));
        for (int i = 0; i < height; i++) {
            image[i] = malloc(width * sizeof(int));
            for (int j = 0; j < width; j++) {
                image[i][j] = rand() % 256;
            }
        }
    }

    int local_new_height = new_height / size; 
    int **local_new_image = calloc(local_new_height, sizeof(int*));
    for (int i = 0; i < local_new_height; i++) {
        local_new_image[i] = calloc(new_width, sizeof(int));
    }

    int *image_linear = NULL;
    if (rank == 0) {
        image_linear = malloc(height * width * sizeof(int));
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                image_linear[i * width + j] = image[i][j];
            }
        }
    }

    int *local_image = malloc((height / size) * width * sizeof(int));
    MPI_Scatter(image_linear, (height / size) * width, MPI_INT,
                local_image, (height / size) * width, MPI_INT,
                0, MPI_COMM_WORLD);

    int thread_count = 4;

    ThreadArgs thread_args[thread_count];
    pthread_t threads[thread_count];

    for (int t = 0; t < thread_count; t++) {
        thread_args[t].thread_id = t;
        thread_args[t].thread_count = thread_count;
        thread_args[t].data = local_image;
        thread_args[t].height = height / size;
        thread_args[t].width = width;
        thread_args[t].output = local_new_image;
        thread_args[t].new_height = local_new_height;
        thread_args[t].new_width = new_width;

        pthread_create(&threads[t], NULL, lanczos_thread_function, (void*)&thread_args[t]);
    }

    for (int t = 0; t < thread_count; t++) {
        pthread_join(threads[t], NULL);
    }

    int *new_image_linear = NULL;
    if (rank == 0) {
        new_image_linear = malloc(new_height * new_width * sizeof(int));
    }

    int *local_new_image_linear = malloc(local_new_height * new_width * sizeof(int));
    for (int i = 0; i < local_new_height; i++) {
        for (int j = 0; j < new_width; j++) {
            local_new_image_linear[i * new_width + j] = local_new_image[i][j];
        }
    }

    MPI_Gather(local_new_image_linear, local_new_height * new_width, MPI_INT,
               new_image_linear, local_new_height * new_width, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        new_image = malloc(new_height * sizeof(int*));
        for (int i = 0; i < new_height; i++) {
            new_image[i] = malloc(new_width * sizeof(int));
            for (int j = 0; j < new_width; j++) {
                new_image[i][j] = new_image_linear[i * new_width + j];
            }
        }

        write_png("output.png", new_image, new_width, new_height);
    }

    MPI_Finalize();
    return 0;
}
