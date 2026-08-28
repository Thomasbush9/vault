# Strings 


```{c}
char *s = "Hello";

printf("%s\n", s);
```

The string variable is just a pointer to the first character of s


We can use the array init char s[] = ... 

That means that we can print it using the array notation 
printf("%c", s[i]);

Since string literals (s=) are placed in a different memory form the rest of the script, you can't change it (mutate), to be able to do so you have to define it using the array notation. 

To get the string length you have to use the string.h function strlen(s) -> bytes. 

In C a string is defined by two basic characteristics: 

-  A pointer to the first character
- A zero-valued byte (NULL char) somewhere in memory after the pointer that indicates the end of the string. 

This because you save a byte by saving string like this as opposed to store the bytes of the string along with a number indicating the length of the string. 

So when you define a string you (implicitly) include the null value \0
char *s = "Hello"-> HELLO\0

Thus, you can count the length of a string by doing: 

```{c}
int str_leng(char *s)
{
  int count = 0;

  while (s[count] != \0)
    count++;
  return count 
}
```

You can't directly copy strings using = (as you'd just copy the pointer to the same value), you have to copy byte at the time. 
You should use the function strcpy(dest, src)
