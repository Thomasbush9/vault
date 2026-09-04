# Typedef: Making New Types

Basically we can take an existing type and make an alias: 

```{c}
typedef int antlepe;

antelope x = 10;
```

It follows regular scoping rules, we'd like to put it global so that all functions can use it. 

```{c}
struct animal {
        char *name;
        int leng_count, speed;
    }

typedef struct animal animal;

struct animal y;
animal y;
```
In this case it's useful just to avoid typing "struct" all the time. 

Most of the times, it is done like this: 

```{c}
typedef struct animal {
        char *name;
        int leg_count, speed;
    } animal;

struct animal y;
animal y;
```

Even more concise: 


```{c}
typedef struct {
        char *name;
        int leg_count, speed;
    } animal;
animal y;
```

a cool function of typedef is that it allows us to change types of variables all at once: 

```{c}
typedef float type_float;

\\ here we can simply do: 
typedef int type_float;

type_floar a1, a2, a3;
```


