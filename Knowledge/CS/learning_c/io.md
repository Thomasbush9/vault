# File Input Outputs


We operate over a piece of data in the form of FILE* type. 
This holds all the information needed to communicate with the I/O 
subsystem. 

There are three types of streams: 

1. standard input 
2. standard output 
3. standard error

In a POSIX shell you can run a program, and send non-error (stdout) to one file and the 
error (stderror) to another file: 

```{bash}
./foo > output.txt 2> errors.txt
```

### Reading Text Files 

Streams are usually divided in: **text** and **binary**

EOF:

it is a special character that is returned from fgetc()
when the end of the file has been reached. 

We can use it to understand when we are at the end of the file

```{c}
#include<stdio.h>

int main(void)
{
        FILE *fp;
        int c;

        fp = fopen("hello.txt", "r");
        while ((c==fgetc(fp)) != EOF)
            printf("%c", c);
        fclose(fp);
    }
```

We are still reading one char at the time, we can read lines with: 

```{c}
#include <stdio.h>

int main(void){
        FILE *fp; 
        char s[1024]; \\ big enough to contain any line 
        int linecount = 0;

        fp = fopen("quote.txt", "r");
        while(fgets(s, sizeof s, fp) != NULL)
            printf("%d: %s", ++linecount, s);
        fclose(fp);
    }

```
fgets: takes three arguements: pointer to a char buffer to hold bytes, max number of byter to read, file

it returns NULL on end-of-file error.


### Formatted Input: 

We can use fscanf() 

```{c} fscanf(fp, "%s %f %d", name, &len, &mass)`

It returns a EOF when ending the file. 


### Writing text Files

We can use: fputc(), fputs(), fprintf() to write text streams,
we just have to open the file with the flag w. 

```{c}
#include <stdio.h>

int main(void)
{
        FILE *fp;
        int x = 32;

        fp = fopen("output.txt", "w");
        fputc("B", fp);
        fprintf(fp, "x=%d\n", x);
        fputs("HELLO, workd \n", fp);

        fclose(fp);
    }
```


### Binary File I/O


We can load them using "b", functions: fread(), fwrite()

Program to write some byte to disk: 

```{c}
#include <stdio.h>

int main(void)
{
        FILE *fp;
        unsigned char bytes[6]= {5, 37, 0, 88, 255, 12};
        fp = fopen("outout.bin", "wb")

        \\pointer to data to write, size of each el, count, file 
        fwrite(bytes, sizeof(char), 6, fp)
    }
```







