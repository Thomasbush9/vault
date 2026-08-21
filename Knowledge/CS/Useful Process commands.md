You can: 

- ss -tunlp to identify processed and users by port 
- lsof -i :portnumber to identiy processed by port number 
- netstat -natup list network owning proceses 

You can check where they are running: ps -fp pid 
and stop them by kill pid 

You can also check the live activity with: sudo cat /proc/pid/environ | tr '\0' '\n'

lsof -p pid -> check open files 

