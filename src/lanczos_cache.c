#include <stdio.h>
#include <math.h>
#include <png.h>
#include <stdlib.h>
#include <string.h>

#define LANCZOS_RADIUS 3
#define INITIAL_SIZE 512
#define FINAL_SIZE 4096
#define TILE_SIZE 32 // Optimal tile size for cache

void write_png(const char *filename, int *matrix, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        perror("File opening failed");
        return;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
    {
        fclose(fp);
        fprintf(stderr, "Failed to create png struct\n");
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        fprintf(stderr, "Failed to create info struct\n");
        return;
    }

    if (setjmp(png_jmpbuf(png)))
    {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        fprintf(stderr, "Error during png creation\n");
        return;
    }

    png_init_io(png, fp);

    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    for (int y = 0; y < height; y++)
    {
        png_bytep row = (png_bytep)malloc(width * 4 * sizeof(png_byte)); // 4 bytes per pixel for RGBA
        for (int x = 0; x < width; x++)
        {
            int grayscale = matrix[y * width + x];
            grayscale = grayscale > 255 ? 255 : (grayscale < 0 ? 0 : grayscale); // Clamp to [0, 255]

            row[x * 4 + 0] = grayscale; // Red
            row[x * 4 + 1] = grayscale; // Green
            row[x * 4 + 2] = grayscale; // Blue
            row[x * 4 + 3] = 255;       // Alpha
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
        return 1.0;
    }
    else if (x >= a || x <= -a) {
        return 0.0;
    }
    else {
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
            if (i < 0 || i >= height || j < 0 || j >= width)
                continue;

            double weight = lanczos_kernel(x - i) * lanczos_kernel(y - j);
            result += weight * data[i * width + j];
            weight_sum += weight;
        }
    }

    return (weight_sum != 0) ? (int)(result / weight_sum) : 0;
}

void apply_2d_lanczos(int *data, int height, int width, int *output, int new_height, int new_width) {
    double scale_height = (double)(height - 1) / (new_height - 1);
    double scale_width = (double)(width - 1) / (new_width - 1);

    for (int i_block = 0; i_block < new_height; i_block += TILE_SIZE) {
        for (int j_block = 0; j_block < new_width; j_block += TILE_SIZE) {
            for (int i = i_block; i < i_block + TILE_SIZE && i < new_height; i++) {
                for (int j = j_block; j < j_block + TILE_SIZE && j < new_width; j++) {
                    double x = i * scale_height;
                    double y = j * scale_width;
                    int result = lanczos_2d_interpolate(data, height, width, x, y);
                    output[i * new_width + j] = fmax(fmin(result, 255), 0);
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <image_size>\n", argv[0]);
        return 1;
    }

    int width = atoi(argv[1]), height = atoi(argv[1]);
    int *image = malloc(height * width * sizeof(int));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image[i * width + j] = rand() % 256;
        }
    }

    int new_width = atoi(argv[2]), new_height = atoi(argv[2]);
    int *new_image = calloc(new_height * new_width, sizeof(int));

    apply_2d_lanczos(image, height, width, new_image, new_height, new_width);

    // write_png("big_rgba.png", new_image, new_width, new_height);

    // free(image);
    // free(new_image);

    return 0;
}
