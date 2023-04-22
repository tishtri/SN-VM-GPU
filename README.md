## `SN-VM-GPU`: Sleptsov Net VM on GPU

# Sleptsov Net Virtual Machine on Graphics Processing Unit


How to use `SN-VM-GPU` as a part of experimental `SNC IDE&VM`:
--------------------------------------------------------------

We list references to components in "Compatibility" section.

1) Use `Tina` `nd` as graphical editor and its labels with special syntax (section "Transition substitution label") to specify transition substitution of `HSN`.

2) Use `NDRtoSN` to convert `NDR` file of `Tina` into `HSN` or `LSN`. 

3) Use `HSNtoLSN` to compile and link HSN file and mentioned in it `LSN` files into a single `LSN` file.

4) Run `LSN` file on `SN-VM` or `SN-VM-GPU`.


Compatibility: 
-------------- 

`Tina`, `nd`, and `NDR` file format according to https://projects.laas.fr/tina/index.php

`NDRtoSN` and ransition substitution labels according to https://github.com/dazeorgacm/NDRtoSN

`SN-VM` and `LSN` file format according to https://github.com/zhangq9919/Sleptsov-net-processor

`HSNtoLSN` and `HSN` file format according to https://github.com/HfZhao1998/Compiler-and-Linker-of-Sleptsov-net-Program

`SN-VM-GPU` and `MSN` file format according to https://github.com/tishtri/SN-VM-GPU


Command line format: 
-------------------- 

   >sn-vn-gpu < sn_raw_matrix_file > sn_final_marking
   
   Optional parameters of command line: 
    	argv1 -- debug level (defauld 0 -- no debug info printed).
   	
   We recommend to run in multiuser mode to avoid GPU timeout halts.
   

How to compile and build:
-------------------------

Examples of command lines to build and run program are specified as comments within source files. Because of absence of entire grid synchronization facilities within CUDA GPU architecture 35, two variants of source code are provided:

	`sn-vm-gpu-1b.c` -- uses a single block only with a single kernel program;
	`sn-vm-gpu-fk.c` -- uses variable number of blocks with a few kerkels programs.
   

Format of file:
---------------

Sleptsov Net Raw Matrix File Format `MSN` as the net dimension followed by matrices B, D, R, and a vector mu. 

`sn_raw_matrix_file`::

m n
B
D
R
mu

	m -- number of places;
	n -- number of transitions;
	B -- matrix of incoming arcs of transitions;
	D -- matrix of outgoing arcs of transitions;
	R -- matrix of transitive closure of priority arcs;
	mu -- initial marking.
	
We recommend to separate matrices/vectors by a blank row.

Raw matrix file can be obtained from LSN by sn-vm (https://github.com/zhangq9919/Sleptsov-net-processor) with flag -rm. 

The following examples of raw matrix files are enclosed: add.msn, mul.msn, div.msn, d2.msn, d3.msn, d4.msn for addition, multiplication, and division [1,2,3] and exact double exponent counters [5], respectively. 

     
Examples of command lines: 
-------------------------- 

   >sn-vn-gpu < div.msn
   
   >sn-vn-gpu < d3.msn > d3-final-mu.txt
   
   
References: 
----------- 
1. Zaitsev D.A. Sleptsov Nets Run Fast, IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2016, Vol. 46, No. 5, 682 - 693. http://dx.doi.org/10.1109/TSMC.2015.2444414

2. Zaitsev D.A., Jürjens J. Programming in the Sleptsov net language for systems control, Advances in Mechanical Engineering, 2016, Vol. 8(4), 1-11. https://doi.org/10.1177%2F1687814016640159

3. Tatiana R. Shmeleva, Jan W. Owsiński, Abdulmalik Ahmad Lawan (2021) Deep learning on Sleptsov nets, International Journal of Parallel, Emergent and Distributed Systems, 36:6, 535-548, https://doi.org/10.1080/17445760.2021.1945055

4. Dmitry A. Zaitsev, Strong Sleptsov nets are Turing complete, Information Sciences, Volume 621, 2023, 172-182. https://doi.org/10.1016/j.ins.2022.11.098

5. Dmitry A. Zaitsev, MengChu Zhou, From strong to exact Petri net computers, International Journal of Parallel, Emergent and Distributed Systems, 37(2), 2022, 167-186. https://doi.org/10.1080/17445760.2021.1991340

6. Qing Zhang, Ding Liu, Yifan Hou, Sleptsov Net Processor, International Conference ”Problems of Infocommunications. Science and Technology” (PICST2022), 10-12 October, 2022, Kyiv, Ukraine.

7. Hongfei Zhao, Ding Liu, Yifan Hou, Compiler and Linker of Sleptsov Net Program,International Conference ”Problems of Infocommunications. Science and Technology” (PICST2022), 10-12 October, 2022, Kyiv, Ukraine.

----------------------------------------------------------------------- 
@ 2023 Tatiana R. Shmeleva: ta.arta@gmail.com
