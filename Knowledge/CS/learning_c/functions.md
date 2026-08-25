# Functions 

In C, functions must declare both arguments and return types: 

```{c}
#inlcude <stdio.h>

int plus_one(int n)
{
  return n + 1
}
```

It is important to understand that here n is a parameter that copies the argument

The int main(void) is a function that returns an int and accept no arg (you can use void to define functions that return no arg.)

**Function Prototypes:**

You can actually call a function before define it if you use the prototype. 
You can call the prototype by define the function return type, arg and a semicolon. 

```{c}
int foo(void);

int main(void)
{
  int i;
  i = foo(i)
}

int foo(void)
{
  return 4333
}
```




