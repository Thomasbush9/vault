# Arrays

We can declare an array like this: 

```{c}
#include<stdio.h>

int main(void)
{
  int i;
  float f[4]

  f[0] = 3.1
}
```


You can't get the length of an array directly, you have to get the sizeof the array and then divide it by the size of each element of the array. Example, if an int is 4 bytes, array 32 bytes long-> len = 32/4 

**Array Initializers**

You can initialize the arrays ad definition time:

```
int a[5] = {1, 2, }
```

You can't initialize more elements than what you define, less is okay as the others will be filled by 0. 

```{c}
int a[100] = {0};
```

You can also define constants as #define COUNT 5 

**Multidimensional Arrays**

You can define them with :

```{c}
int a[5][5]
```

They are stored in memory in row-major order. 

**Array and Pointers**

Generally speaking when you refer to a pointer of an array you refer to the pointer of the first element of the array. (p=a)


When you pass an array to a function you are actually passing the pointer to the first element and then you can still access to the elements using array notation. 

C needs only the shape after the first one to understand how to access to the right elements in memory.
