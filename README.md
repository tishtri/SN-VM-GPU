## `SN-VM-GPU-MCC v.2.1.MCC`: Sleptsov Net VM on GPU using Matrix with Condensed Columns Format

# Sleptsov Net Virtual Machine on Graphics Processing Unit using Matrix with Condensed Columns Format (MCC) 


## Enhances performance!
## Considerably reduces data size and number of required GPU threads!


New features:
-------------

1) MCC -- an ad-hoc data structure to run efficiently sparse data on GPU;

2) Fast firing transition choice based on transition reordering according to the priority lattice and the first fireable transition choice.


How to use `SN-VM-GPU-MCC` as a part of experimental `SNC IDE&VM`:
------------------------------------------------------------------

We list references to components in "Compatibility" section.

1) Use `Tina`, `nd` as graphical editor and its labels with special syntax (section "Transition substitution label") to specify transition substitution of `HSN`.

2) Use `NDRtoSN` to convert `NDR` file of `Tina` into `HSN` or `LSN`. 

3) Use `HSNtoLSN` to compile and link HSN file and mentioned in it `LSN` files into a single `LSN` file.

4) Run `LSN` file on `SN-VM` 

5) Run `MCC` file on `SN-VM-GPU-MCC`.


Compatibility: 
-------------- 

`Tina`, `nd`, and `NDR` file format according to https://projects.laas.fr/tina/index.php

`NDRtoSN` format conversion providing transition substitution labels according to https://github.com/dazeorgacm/NDRtoSN

`SN-VM` and `LSN` file format according to https://github.com/zhangq9919/Sleptsov-net-processor

`HSNtoLSN` and `HSN` file format according to https://github.com/HfZhao1998/Compiler-and-Linker-of-Sleptsov-net-Program

`SN-VM-GPU` and `MSN` file format according to Release v.1.1 https://github.com/tishtri/SN-VM-GPU

`SN-VM-GPU-MCC` and `MCC` file format according to Release v.2.1.MCC https://github.com/tishtri/SN-VM-GPU


Command line format: 
-------------------- 

I. Run VM:

   >sn-vn-gpu-mcc < sn_mcc_file > sn_final_marking
   
   Optional parameters of command line: 
    	argv1 -- debug level (default 0 -- no debug info printed).
    	argv2 -- limitation of number of SN steps  (no limitation by default).
   	
   We recommend to run in multiuser mode  of Linux/UNIX (without GUI) to avoid GPU timeout halts.
   
II. Convert Raw matrix file (.msn) to Matrix with Condensed Columns file (.mcc):

   >msn-to-mcc < sn_mcc_file > sn_mcc_file
   

How to compile and build:
-------------------------

Examples of command lines to build and run program are specified as comments within source files. Because of absence of entire grid synchronization facilities within CUDA GPU architecture 35, we use a few kernels. Two programs are supplied:

	`sn-vm-gpu-mcc.c` -- uses variable number of blocks with a few kernel programs;
	`msn-to-mcc` -- converts Raw matrix file (.msn) to Sparse matrix file (.mcc).
   

Format of file:
---------------

I. Sleptsov Net Raw Matrix File Format `MSN` as the net dimension followed by matrices B, D, R, and a vector mu. 

`sn_raw_matrix_file`::

m n
B
D
R
mu

	m -- number of places;
	n -- number of transitions;
	B -- matrix of incoming arcs of transitions (m by n);
	D -- matrix of outgoing arcs of transitions (m by n);
	R -- matrix of priority arcs (n by n);
	mu -- initial marking.
	
We recommend to separate matrices/vectors by a blank row.

Raw matrix file can be obtained from LSN by sn-vm (https://github.com/zhangq9919/Sleptsov-net-processor) with flag -rm. 

The following examples of raw matrix files are enclosed: add.msn, mul.msn, div.msn, d2.msn, d3.msn, d4.msn for addition, multiplication, and division [1,2,3] and exact double exponent counters [5], respectively. 


II. Sleptsov Net sparse data based on Matrix with Condensed Columns Format `MCC` as the net dimension, maximal numbers of nonzero elements in a column of matrices B and D followed by matrices BS, DS, and a vector mu; transitions are sorted by priorities to fire the first fireable transition.

`sn_mcc_file`::

m n mm
B-index
B-value
D-index
D-value
mu

	m -- number of places;
	n -- number of transitions;
	mm -- maximal number of nonzero elements in a column of matrix B and D;
	B-index -- matrix of incoming arcs of transitions -- numbers of places for incoming arcs (mm by n);
	B-value -- matrix of incoming arcs of transitions -- multiplicities of incoming arcs (mm by n);
	D-index -- matrix of outgoing arcs of transitions -- numbers of places for outgoing arcs (mm by n);
	D-value -- matrix of outgoing arcs of transitions -- multiplicities of outgoing arcs (mm by n);
	mu -- initial marking.
	
In each column of B* and D* nonzero elements of the corresponding column of B and D are listed, started from beginning and followed by zeroes (when the actual number of nonzero elements within the column is less than mm). Since indices in B-index and D-index can be zero, we use B-value and D-value to find nonzero elements of a matrix column.
	
We recommend to separate matrices/vectors by a blank row.

Sparse matrix file `MCC` can be obtained from Raw matrix file `MSN` by msn-to-mcc. 

The following examples of raw matrix files are enclosed: add.mcc, mul.mcc, div.mcc, d2.mcc, d3.mcc, d4.mcc for addition, multiplication, and division [1,2,3] and exact double exponent counters [5], respectively. 

     
Examples of command lines: 
-------------------------- 

   >sn-vn-gpu-mcc < div.mcc
   
   >sn-vn-gpu-mcc < d3.mcc > d3-final-mu.txt
   
   
References: 
----------- 
1. Zaitsev D.A. Sleptsov Nets Run Fast, IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2016, Vol. 46, No. 5, 682 - 693. http://dx.doi.org/10.1109/TSMC.2015.2444414

2. Zaitsev D.A., Jürjens J. Programming in the Sleptsov net language for systems control, Advances in Mechanical Engineering, 2016, Vol. 8(4), 1-11. https://doi.org/10.1177%2F1687814016640159

3. Dmitry A. Zaitsev, Tatiana R. Shmeleva, Qing Zhang, and Hongfei Zhao, Virtual Machine and Integrated Developer Environment for Sleptsov Net Computing Parallel Processing Letters, Vol. 33, No. 03, 2350006 (2023). https://doi.org/10.1142/S0129626423500068

4. Tatiana R. Shmeleva, Jan W. Owsiński, Abdulmalik Ahmad Lawan (2021) Deep learning on Sleptsov nets, International Journal of Parallel, Emergent and Distributed Systems, 36:6, 535-548, https://doi.org/10.1080/17445760.2021.1945055

5. Dmitry A. Zaitsev, Strong Sleptsov nets are Turing complete, Information Sciences, Volume 621, 2023, 172-182. https://doi.org/10.1016/j.ins.2022.11.098

6. Bernard Berthomieu, Dmitry A. Zaitsev, Sleptsov Nets are Turing-complete, Theoretical Computer Science, Volume 986, 2024, 114346. https://doi.org/10.1016/j.tcs.2023.114346

7. Dmitry A. Zaitsev, MengChu Zhou, From strong to exact Petri net computers, International Journal of Parallel, Emergent and Distributed Systems, 37(2), 2022, 167-186. https://doi.org/10.1080/17445760.2021.1991340

--------------------------------------------------------------------------------------------------------------- 
@ 2024 Tatiana R. Shmeleva: ta.arta@gmail.com
