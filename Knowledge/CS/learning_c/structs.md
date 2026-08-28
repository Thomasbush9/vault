# Structs 

They are basically a class with only data members and not methods. 

```{c}
struct cat {
  char *name;
  float price;
  int speed;
}
// initalize the class
struct car sat = {.speed=15, };
// now the rest is 0 
//access 
sat.speed 
```


When you pass it to a function you can: pass struct copy or the pointer to it. Usually you want to pass the pointer (save memory + you can change it) 

To change it you have to use the arrow operator:

```{c}
void set_price(struct car *c, float new_price) {
  //(*c).price = new_price (we dereference array)

  c->price=new_price 
}
```

To compare structs you have to iterate element by element


