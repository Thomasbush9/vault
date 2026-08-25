# Variables

Each value is stored in memory, we can refer to it by using the address (pointer) on the memory array, or use the variable name. 

Before you can use a variable you have to declare that variable and what type it can hold. Once it is declared it cannot be change type at runtime. 

### Operations

here I summarize some of the operation that are more original to C. 

Ternary operation: 

y += x>10? 17: 37 

it means that we add 17 to y if x is more than 10 otherwise we add 37

These operators don't have flow control like an if statement, they just evaluate a value. 

Post/pre increments:

i++ increment i of 1 after it has been evaluated 
++i adds 1 to i before evaluation 

Example:

5+i++ -> first computes 5+i and then adds 1 to i. 


Boolean operators:

&& and 
|| or 
! not 

Size of: 
sizeof returns the size (bytes) that a particular variable uses in memory. 
it has a special type size_t 

Flow controls:

They have the basics + a do-while that executes the while loop 1 time before eval. 

do{
  command 
} while (i<10)

Random:

#inlcude <stdlib.h>

rand()%range

Switch statement: 

evaluate the expression based on the based

int main(void)
{
int goat_count = 2;

switch (goat_count) {
case 0:
printf("You have no goats.\n");
 break;

 case 1:
 printf("You have a singular goat.\n");
 break;

 case 2:
 printf("You have a brace of goats.\n");
 break;

 default:
 printf("You have a bona fide plethora of goats!\n");
 break;
}


