#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    //allocate space for 20 floats 
    float *p = malloc(sizeof *p *20);

    // assign them to fractions: 
    for (int i=0; i<20; i++)
        p[i] = 1 / 20.0;
    //change the array to contains 40 elements 
    float *new_p = realloc(p, sizeof *p *40);

    // check if it allocated: 
    if (new_p == NULL){
        printf("Error reallocating\n");
        return 1;
    }

    p = new_p;

    //assign new elements: 
    for (int i=20; i<40; i++)
        p[i] = 1.0 + (i-20) / 20.0;

    // print values 
    for (int i=0; i<40; i++)
        printf("%f\n", p[i]);

    free(p);


}
