How containers are implemented? 
By looking at the linux kernel source code: containers are not present in the kernel source code. 
In the kernel we can find: 
cgroups: needed to set limits on the runs-> each subsystem (cpu, memory) has its own hierarchy -> each process is in a node in each hierarchy. Basically each hierarchy starts with 1 node (root)-> node = group of processes. 
Memory cgroup:
	- accounting: keeps track of pages used by each group (anonymous, active and inactive memories), each page is charged to a group, groups can shared the 
	- Limits: each group can have itw own limits: soft limits (not enforced): , Hard: process kill if it goes above limit (OOM). You can set limits for different kinds of memory (physical, kernel)-> we can avoid OOM with a warning to move the container to a different machine, add more memory 
- HugeTLB cgroup: limit the amount of memory huge pages per process
- CPU usage: keep tracks of user/system cpu time, track of usage per cpu. You can't set limits-> if you are using only a small amount of cpu percentage, then modern machines would slow down the cpu (look more into that)
- Cpuset cgroup: pin groups to specific cpus: reserve cpu for specific task (process)
- Blkio cgroup: keep track of I/O for each group: both for network and disk. 
- Devices cgroups: controls what the group can do on device nodes 

Namespaces: they limit what you can view
Each process is in one namespace of each type:
- pid: processes within a pid namespace only see 
- Network: processes within a given network namespace get their own private network stack 
- mnt: mount things to the container: they can have private mounts 
- ipc: allows ipc resources to processes 
- user namespace: allow to map UID 
- namespace manipulation: when you create a new process you pass its namespaces-> namespaces are materialized by pseudo-files 

Copy-on-Write: 
create a new container on the fly. 

Container runtimes: based on cgroups + namespaces 

Exercise: build a small container-building script in go or bash? 