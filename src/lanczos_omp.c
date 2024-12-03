#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define LANCZOS_RADIUS 3

#pragma region NO_PARRALLEL

void print(int **a, int height, int width)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
}

#pragma endregion

double lanczos_kernel(double x)
{
    int a = LANCZOS_RADIUS;
    if (x == 0) {
        return 1.0; // sinc(0) = 1
    }
    else if (x == a || x == -a) {
        return 0.0; // Lanczos function for values of x = ±a
    }
    else {
        return (sin(M_PI * x) / (M_PI * x)) * (sin(M_PI * x / a) / (M_PI * x / a));
    }
}

int lanczos_2d_interpolate(int **data, int height, int width, double x, double y)
{
    int a = LANCZOS_RADIUS;

    double result = 0;
    double weight_sum = 0;

    int center_x = (int)x;
    int center_y = (int)y;

// #pragma omp parallel for collapse(2) reduction(+ : result, weight_sum) -- mai lent
    for (int i = center_x - a + 1; i < center_x + a; i++)
    {
        for (int j = center_y - a + 1; j < center_y + a; j++)
        {
            if (i < 0 || i >= height || j < 0 || j >= width)
            {
                continue;
            }
            double weight = lanczos_kernel(x - i) * lanczos_kernel(y - j);
            result += weight * data[i][j];
            weight_sum += weight;
        }
    }

    if (weight_sum != 0)
    {
        result /= weight_sum;
    }

    return result;
}

void apply_2d_lanczos(int **data, int height, int width, int **output, int new_height, int new_width)
{
    int a = LANCZOS_RADIUS;
    double scale_height = (double)(height - 1) / (new_height - 1);
    double scale_width = (double)(width - 1) / (new_width - 1);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < new_height; i++)
    {
        for (int j = 0; j < new_width; j++)
        {
            double x = i * scale_height;
            double y = j * scale_width;
            int result = lanczos_2d_interpolate(data, height, width, x, y);
            result = fmax(fmin(result, 255), 0);
            output[i][j] = result;
        }
    }
}

#pragma region DEPRECATED

// DEPRECATED
int lanczos_1d_interpolate(int *data, int length, double x)
{
    int a = LANCZOS_RADIUS;
    int i;
    double result = 0.0;
    int center = (int)x; // the integer part of x
    double sum = 0.0;    // normalization factor (sum of kernels)

    // Sum up the values weighted by the Lanczos kernel
    for (i = -a + 1; i < a; i++)
    {
        if (center + i < 0 || center + i >= length)
            continue;
        double weight = lanczos_kernel(x - (center + i));
        result += weight * data[center + i];
        sum += weight;
    }

    // Normalize the result by the sum of the weights
    if (sum != 0)
    {
        result /= sum;
    }

    return result;
}

// DEPRECATED
void apply_1d_lanczos(int *data, int length, int *output, int output_length)
{
    int a = LANCZOS_RADIUS;
    double scale = (double)(length - 1) / (output_length - 1); // scaling factor for resampling
    for (int i = 0; i < output_length; i++)
    {
        double x = i * scale;
        output[i] = lanczos_1d_interpolate(data, length, x);
    }
}

#pragma endregion

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <initial_size> <final_size>\n", argv[0]);
        return 1;
    }

    int width, height;
    width = atoi(argv[1]);
    height = atoi(argv[1]);

    int **image = malloc(height * sizeof(int *));
    for (int i = 0; i < height; i++)
    {
        image[i] = malloc(width * sizeof(int));
        for (int j = 0; j < width; j++)
        {
            image[i][j] = rand() % 256;
        }
    }

    int new_width = atoi(argv[2]);
    int new_height = atoi(argv[2]);

    int **new_image = calloc(new_height, sizeof(int *));
    for (int i = 0; i < new_height; i++)
        new_image[i] = calloc(new_width, sizeof(int));

    apply_2d_lanczos(image, height, width, new_image, new_height, new_width);

    // printf("Original image:\n");
    // print(image, height, width);

    // FILE *original_file = fopen("original.txt", "w");
    // if (original_file != NULL)
    // {
    //     fprintf(original_file, "%d %d\n", height, width);
    //     for (int i = 0; i < height; i++)
    //     {
    //         for (int j = 0; j < width; j++)
    //         {
    //             fprintf(original_file, "%d ", image[i][j]);
    //         }
    //         fprintf(original_file, "\n");
    //     }
    //     fclose(original_file);
    // }
    // else
    // {
    //     perror("Failed to open file for writing");
    // }

    // FILE *resampled_file = fopen("resampled.txt", "w");
    // if (resampled_file != NULL)
    // {
    //     fprintf(resampled_file, "%d %d\n", new_height, new_width);
    //     for (int i = 0; i < new_height; i++)
    //     {
    //         for (int j = 0; j < new_width; j++)
    //         {
    //             fprintf(resampled_file, "%d ", new_image[i][j]);
    //         }
    //         fprintf(resampled_file, "\n");
    //     }
    //     fclose(resampled_file);
    // }
    // else
    // {
    //     perror("Failed to open file for writing");
    // }
    // printf("Resampled image:\n");
    // print(new_image, new_height, new_width);

    // write_png("small_rgba.png", image, width, height);
    // write_png("big_rgba.png", new_image, new_width, new_height);

    return 0;
}
