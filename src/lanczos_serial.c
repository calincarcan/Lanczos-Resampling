// Calin

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define LANCZOS_RADIUS 3

void print(int **a, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
}

double lanczos_kernel(double x)
{
    int a = LANCZOS_RADIUS;
    if (x == 0)
    {
        return 1.0;
    }
    else if (x >= a || x <= -a)
    {
        return 0.0;
    }
    else
    {
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

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <initial_size> <final_size>\n", argv[0]);
        return 1;
    }
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
