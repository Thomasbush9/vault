# Pointer Arithmetic 


If you have a pointer in memory, adding 1 to it moves the pointer to the next item of that type directly after in memory. 

**It is important to be sure that we are pointing to a valid place in memory before we derefence**

==Adding to Pointers==

Check the script on pointer addition. 

If we add 1 to pointer, it actually jumps of sizeof(int) bytes to the next int.
It knows to jump that far as it is an int pointer. 


==Subtracting Pointers==

You can subtract two pointers to find teh difference between them:
how many ints there are between two int*s. However, this only works 
for a single array. 

==Array pointer equivalance==

We can say: 

a[b] = *(a+b)

We can use array or pointer operations for any array/pointer. 

This concept is mostly encoutered in function calls: 

```{c}
int my_strlen(char *s)

// you can pass either an arrray or a pointer 

char s[] = "Antelopes";
char *t = "Wombats";
```


==Void Pointers==


Using void* means that we have a pointer to something that we don't know the type of. 
There are two cases to use it:

- A function is gong to operate on something byte-by-byte:
    memcpy() copies bytes of memory from one pointer to another (they can have any type)

- Another function is calling a function you pased to it and it's passing you data.
    

There are some limits with void pointers:

1. You cannot do pointer Arithmetic
2. You cannot dereference a void*
3. You cannot use the arrow operatore on a void*
4. You cannot use array notation on a void*

Because all these operations require you to know the size of the elements of the pointer. 

You can convert void* to a type before using it (to derefence if needed):

```{c}
char a = "x";

void *p = &a;
char *q = p; 

printf("%c\n", p) //error!
printf("%c\n", q) //Allowed
```

We can write our own mmcpy()

```{c}
void *my_memcpy(void *dest, void *src, int byte_count)
{
        //convert void *s to chars *s 
        char *s, *d = dest;
        while (byte_count--){
                *d++ = *s++;
            }
        return dest;
    }
```

Function qsort()-> it can sort anything:


