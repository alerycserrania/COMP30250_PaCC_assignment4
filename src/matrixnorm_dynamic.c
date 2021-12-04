#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>


void print_matrix(int N, double* M)
{
    int i, j, k;

    for (i = 0; i < N; i++) 
    {
        for (j = 0; j < N; j++) 
        {
            printf("%f\t", M[i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int m_size, nb_thrds, slice_size;
    double *a, *b, *c;
    double norm, sum;
    int i, j, k;
    int chunk_size;
    
    struct timeval tv1, tv2;
    struct timezone tz;
    double elapsed;

    char PRINT_RESULT, PRINT_TIME;

    if(argc != 5)
    {
       printf("Please, use: %s N C PRINT_RESULT PRINT_TIME:\n", argv[0]);
       printf("\t- N: matrix size\n");
       printf("\t- C: chunk size\n");
       printf("\t- PRINT_RESULT (y/n): print result to stdout\n");
       printf("\t- PRINT_TIME (y/n): print time elasped to stdout\n");
       exit(EXIT_FAILURE);
    }

    m_size = atoi(argv[1]);
    chunk_size = atoi(argv[2]);
    nb_thrds = omp_get_num_procs();
    PRINT_RESULT = argv[3][0];
    PRINT_TIME = argv[4][0];

    if (m_size < nb_thrds) {
        printf("Number of threads must be less than the size");
        exit(EXIT_FAILURE);
    }

    slice_size = m_size/nb_thrds;

    a = malloc(m_size * m_size * sizeof(double));
    b = malloc(m_size * m_size * sizeof(double));
    c = malloc(m_size * m_size * sizeof(double));

    for (i = 0; i < m_size * m_size; i++)
    {
        a[i] = 1. + i;
        b[i] = 1.;
        c[i] = 0.;
    }

    omp_set_num_threads(nb_thrds);

    gettimeofday(&tv1, &tz);

    // parallel multiplication of matrices
    #pragma omp parallel for schedule(dynamic, chunk_size) shared(a, b, c, m_size) private(sum, j, k)
    for (i = 0; i < m_size; i++)
    {
        for (j = 0; j < m_size; j++)
        {
            sum = 0.0;
            for (k = 0; k < m_size; k++)
            {
                sum += a[i*m_size+k] * b[k*m_size+j];
            }
            c[i*m_size+j] = sum;
        }
    }

    // parallel computation of matrix norm
    norm = 0.0;
    #pragma omp parallel for schedule(dynamic, chunk_size) shared(c, m_size) private(i, sum) reduction(max: norm)
    for (j = 0; j < m_size; j++)
    {
        sum = 0.;
        for (i = 0; i < m_size; i++)
        {
            sum +=  c[i*m_size+j] > 0. ?  c[i*m_size+j] : -c[i*m_size+j];
        }

        norm = (norm < sum) ? sum : norm;
    }

    gettimeofday(&tv2, &tz);


    if (PRINT_RESULT == 'y')
    {
        printf("A =\n");
        print_matrix(m_size, a);
        printf("B =\n");
        print_matrix(m_size, b);
        printf("C =\n");
        print_matrix(m_size, c);
        printf("Norm: %f\n", norm);
    }

    if (PRINT_TIME == 'y')
    {
        elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec)*1e-6;
        printf("elapsed: %fs\n", elapsed);
    }

    free(a);
    free(b);
    free(c);

    return 0;
}

