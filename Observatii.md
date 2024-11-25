# Observatii OMP

Ruland bucata de cod

```c
int lanczos_2d_interpolate(int **data, int height, int width, double x, double y)
{
    int a = LANCZOS_RADIUS;

    double result = 0;
    double weight_sum = 0;

    int center_x = (int)x;
    int center_y = (int)y;

#pragma omp parallel for collapse(2) reduction(+ : result, weight_sum)
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
```

folosind paradigma OMP, timpul de rulare este mai lent, in comparatie cu rularea fara OMP. Acest lucru se datoreaza faptului ca, in cazul de fata, operatiile de adunare si inmultire sunt mai costisitoare decat operatiile de sincronizare si impartire a task-urilor. In cazul in care operatiile de sincronizare si impartire a task-urilor sunt mai costisitoare, atunci OMP ar putea imbunatati performanta.
