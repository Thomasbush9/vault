chmod who op perms target 

WHO:
- u user/owner
- g group 
- o other 
- a all three above
Operator: 
- + add this permission 
- - remove this permission 
- = set exactly this, clears everything that is not listed o=r -> r--

Permissions:
- r read
- w write: modify a file 
- x: execute run a file or traverse it 
- X: conditional execute 
- s: setuid: run as owner/group 
- t: sticky bit: in a shared dir only file's owner cna delete it 

Main flag: 
- R: recursive

Octal shorthand: 
r=4, w=2, x=1 summed per who field in ugo: chmod 750 -> owner rwx (7) group r-x 5 and other 0 