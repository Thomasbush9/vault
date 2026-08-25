# Pointers 

Each byte of memory is identified with an integer. They increase sequentially as 
you move in memory. Each number in the memory array represents an address. 

Thus, a pointer is a variable that hold an address. 

We can use the operator ```address-of` to find the address of the data. 
It can also be replaced by ```&` 

To print it we should use the ```%p` 

Example: 

```{c}
#include<stdio.h>

int main(void)
{
  int i = 10;

  printf("The value of i is %d\n", i);
  printf("And its address is %p\n". (void *)&i);
}
```

Computers use virtual memory to store data-> they think that they have the entire 
memory for themselves.

You can use the pointer type and de-reference a pointer:

```{c}
#include<stdio>

int main(void)
{
  int i;
  int *p;
  
  p=&i; // p points to i 

  i = 10;
  *p = 20 // the thing p points (i) is now 20

}
```


We can pass pointers to functions to access the variables and modify them instead of operating on copies. 

```{c}
#include<stdio.h>

void increment(int *p)
{
  *p = *p +1;
}

int main(void)
{
  int i = 10;
  int *j= &i;

  increment(j);

  printf("i is %d\n", i);

}
```

The NULL pointer: any pointer variable of any pointer type can be set to a special value called NULL-> the pointer does not point to anything. 

```{c}
int *p;
p = NULL;
```

It cannot be de-referenced



