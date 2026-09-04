# Manual Memory Allocation


Other languages use reference counting, garbage collection.
In C you can tell to allocate for you a certain number of bytes that you can use as you please. 
And they remain in memory until you free them. 

Automatic local variables are allocated on the **stack** and manually allocated memory is on the **heap**. 

### Allocating and Deallocating malloc() and free()

Malloc accepts a number of bytes to allocate and returns a void pointer to that block. 
You can use the sizeof() function to decide how many bytes to allocate. 

```{c}
#include <stdlib.h>

int *p = malloc(sizeof(int));
*p = 12;

printf("%d\n", *p);
free(p);
```

You can use sizeof also like this: int *p = malloc(sizeof *p);

### Error Checking 

Malloc returns NULL if it can't allocate memory. 

```{c}
int *x;

x = malloc(sizeof(int)*10);
if (x == NULL){
        printf("Error allocatin 10 ints\n");
    }
```

### Allocating Space for an Array 

We can multiply the size of the thing that we want by the number of elements. 

```{c}
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
        int *p = malloc(sizeof(int)*10);
        for (int i =0; i<10; i++)
            p[i] = i;
        for (int i = 0; i<10; i++)
            prinf("%d\n", p[i]);
        free(p);
    }
```

### Calloc:

It works similarly to Malloc, but: 

- you pass size of one element + num elements 
- it clears the memory to 0 

```{c}
int *p = calloc(10, sizeof(int));
int *q = malloc(sizeof(int)*10);
memset(q, 0, 10*sizeof(int));
```


### Chaning Allocated Sizue with realloc()

It takes a pointer to some previously allocated memory and a new size for teh memory region to be. 

```{c}
num_floats *=2;
np = realloc(p, num_floats); // wrong you need the number of bytes
np = realloc(p, num_floats*sizeof(float));
```



