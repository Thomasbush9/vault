#include "stdio.h"
#include <stdio.h>

int main(void)
{
    int a[5] = {11, 22, 33, 44, 55};

    //get pointer to first element 
    int *p = &a[0];
    // printf("%d\n", *p);
    // // now we can add one to the pointer: 
    // printf("%d\n", *(p + 1));
    //for loop:
    for (int i=0; i<5; i++) {
        printf("%d\n", *(p+i));
    }
}
