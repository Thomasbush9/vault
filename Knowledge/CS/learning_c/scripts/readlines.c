#include <stdio.h>
#include <stdlib.h>


char *readline(FILE *fp)
{
    int offset = 0; // Index next char goes in the buffer
    int bufsize = 4; //initial size power of 2
    char *buf; //the buffer 
    int c; // the character that we have read 
    
    buf = malloc(bufsize);

    if (buf == NULL)
        return NULL;

    //loop read until newline or EOF 
    while (c=fgetc(fp), c != '\n' && c != EOF){
        // check if we're out of room in the buffer accounting
        // for the extra byte for the NULL terminator 
        if (offset == bufsize -1 ){
            bufsize *= 2; // 2x the space 

            char *new_buf = realloc(buf, bufsize);

            if (new_buf == NULL) {
                free(buf);
                return NULL;
            }

            buf = new_buf; // successful realloc
        }

        // offset ++ changes it in place
        buf[offset++] = c; // add the byte onto the buffer
    }

    if (c == EOF && offset==0){
        free(buf);
        return NULL;
    }
    if (offset < bufsize-1) {
        char *new_buf = realloc(buf, offset+1);

        if (new_buf != NULL)
            buf = new_buf;

    }

    buf[offset] = '\0';
    return buf;
}

int main(void)
{
    FILE *fp = fopen("foo.txt", "r");

    char *line;

    while ((line == readline(fp)) != NULL) {
        printf("%s\n", line);
        free(line);
    }
    fclose(fp);
}
