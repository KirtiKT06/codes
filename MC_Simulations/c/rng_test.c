#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() 
{
    int N = 1000000;
    FILE *fp;

    srand(time(NULL));

    fp = fopen("../data/uniform.dat", "w");
    if (fp == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        double u = (double) rand() / RAND_MAX;
        fprintf(fp, "%lf\n", u);
    }

    fclose(fp);
    return 0;
}