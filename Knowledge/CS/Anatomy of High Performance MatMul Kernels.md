![Full article here](https://www.aleksagordic.com/blog/matmul)
Overall in a GPU we have two operations:
- Memory swaps 
- Computations 
### Memory:

Memory is hierarchical: 
- SRAM cells: faster, but larger 
- DRAM cells: slower, but more dense -> more memory.
SRAM is placed close the compute units and then slower memory cells are placed further and further. 
- Device Memory (80Gb in H100): off-chip DRAM, it hosts the shared global memory (GMEN)
- L2 cache: large, k-way set-associateive cache built on top of SRAM: SM connects to only partitions (total 2)
- L1 cache and shared memory: 
- Register file (RMEN): faster storage next to the compute units 
![[Screenshot 2026-03-27 at 20.35.03.png]]

### Compute: 

The fundamental unit is the Streaming MultiProcessor SM: Hopper H100 has 132 SMs in total. They are grounded in graphic processing clusters (GPCs) each containing 18SMs, and there are 8GPCs on the GPU.  Four GPC connects directly on a L2 partition, the other four to the other one. 

