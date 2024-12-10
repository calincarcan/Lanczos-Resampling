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

## Timpi rulare

```md
Time taken for lanczos_serial with 512 and 16384 is 17.96695566177368
Time taken for lanczos_serial with 1024 and 32768 is 73.223708152771
Time taken for lanczos_serial with 2048 and 65536 is 295.1186828613281
Time taken for lanczos_serial with 4096 and 131072 is 1188.6446425914764
```

```md

Time taken for lanczos_serial with 512 and 4096 is 18.404731273651123
Time taken for lanczos_serial with 1024 and 8192 is 72.58419895172119
Time taken for lanczos_serial with 2048 and 16384 is 293.17889428138733
Time taken for lanczos_serial with 4096 and 32768 is 1189.456300020218
Time taken for lanczos_serial with 8192 and 65536 is 1986.3329153060913
```

```md
Time taken for lanczos_omp with 512 and 4096 is 2.3909101486206055
Time taken for lanczos_omp with 1024 and 8192 is 9.383034944534302
Time taken for lanczos_omp with 2048 and 16384 is 37.337162017822266
Time taken for lanczos_omp with 4096 and 32768 is 148.84621214866638
Time taken for lanczos_omp with 8192 and 65536 is 0.9481332302093506 -- eroare de memorie?
```

```md
Time taken for lanczos_pthreads with 512 and 4096 is 2.464463472366333
Time taken for lanczos_pthreads with 1024 and 8192 is 9.779581308364868
Time taken for lanczos_pthreads with 2048 and 16384 is 39.13181805610657
Time taken for lanczos_pthreads with 4096 and 32768 is 155.4556438922882
Time taken for lanczos_pthreads with 8192 and 65536 is 264.1315107345581
```

```md
Time taken for lanczos_memo_serial with 512 and 4096 is 3.887202739715576
Time taken for lanczos_memo_serial with 1024 and 8192 is 15.282569169998169
Time taken for lanczos_memo_serial with 2048 and 16384 is 61.23960781097412
Time taken for lanczos_memo_serial with 4096 and 32768 is 246.86938285827637
Time taken for lanczos_memo_serial with 8192 and 65536 is 419.34121584892273

```
