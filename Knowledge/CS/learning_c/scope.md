# Scope

The general rule is that if you have declared a variable inside a block, it's Scope
is inside that block. 

Example of nested scope: 
```{c}
#include <stdio.h>

int main(void)
{
        int a = 32;

        if (a==12){
                int b=99;
            }
    }
```

If you have a variable named the same thing at an inner scopee, the one at the inner
scope takes precedence. 


**File Scope**

If you define a variable outisde of a block, that variable has file scope. 
It's visible to all functions int he file that come after it. 




