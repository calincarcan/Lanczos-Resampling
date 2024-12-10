#include <stdio.h>
#include <math.h>
#include <png.h>
#include <stdlib.h>

#define LANCZOS_RADIUS 3
#define initial_size 512
#define final_size 4096
#pragma region NO_PARRALLEL

void write_png(const char *filename, int **matrix, int width, int height) {
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
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // Write each row of the matrix to the PNG file as RGBA values
    for (int y = 0; y < height; y++) {
        png_bytep row = (png_bytep)malloc(width * 4 * sizeof(png_byte)); // 4 bytes per pixel for RGBA
        for (int x = 0; x < width; x++) {
            int grayscale = matrix[y][x];
            grayscale = grayscale > 255 ? 255 : (grayscale < 0 ? 0 : grayscale); // Clamp to [0, 255]

            row[x * 4 + 0] = grayscale; // Red
            row[x * 4 + 1] = grayscale; // Green
            row[x * 4 + 2] = grayscale; // Blue
            row[x * 4 + 3] = 255;       // Alpha (fully opaque)
        }
        png_write_row(png, row);
        free(row);
    }

    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

void print(int **a, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
}

#pragma endregion

double lanczos_kernel(double x) {
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

#pragma region DEPRECATED

// DEPRECATED
int lanczos_1d_interpolate(int *data, int length, double x) {
    int a = LANCZOS_RADIUS;
    int i;
    double result = 0.0;
    int center = (int)x; // the integer part of x
    double sum = 0.0;    // normalization factor (sum of kernels)

    // Sum up the values weighted by the Lanczos kernel
    for (i = -a + 1; i < a; i++) {
        if (center + i < 0 || center + i >= length)
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
void apply_1d_lanczos(int *data, int length, int *output, int output_length) {
    int a = LANCZOS_RADIUS;
    double scale = (double)(length - 1) / (output_length - 1); // scaling factor for resampling
    for (int i = 0; i < output_length; i++) {
        double x = i * scale;
        output[i] = lanczos_1d_interpolate(data, length, x);
    }
}

#pragma endregion

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
