// SN-VM-GPU v.2.1.MCC
// Sleptsov net Virtual Machine on GPU with sparse data format .mcc based on ad-hoc
// Matrix with Condensed Columns (MCC) concept
// Enhances performance!
// Considerably reduces data size and number of required GPU threads!
//
// Uses variable number of GPU blocks with a few kernel programs
// Provides compatibility with NVIDIA architecture 35
// Compile: nvcc sn-vm-gpu-mcc.cu -o sn-vm-gpu-mcc -gencode arch=compute_35,code=compute_35 -Wno-deprecated-gpu-targets
// No GPU timeout: sudo systemctl isolate multi-user.target
// Run:     ./sn-vm-gpu-mcc < net.sns
// @ 2024 Tatiana R. Shmeleva: ta.arta@gmail.com

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>

#include <cuda_runtime.h>

#define MATRIX_SIZE(d1,d2,t) ((d1)*(d2)*(sizeof(t)))
#define VECTOR_SIZE(d1,t)    ((d1)*(sizeof(t)))

#define MOFF(i,j,d1,d2) ((d2)*(i)+(j))
#define MELT(x,i,j,d1,d2) (*((x)+MOFF(i,j,d1,d2)))

#define zmax(x,y) (((x)>(y))?(x):(y))
#define zmin(x,y) (((x)<(y))?(x):(y))

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void zero_matr(int *x,int m,int n)
{
  memset(x,0,MATRIX_SIZE(m,n,int));
}

void zero_mu(int *x,int m)
{
  memset(x,0,VECTOR_SIZE(m,int)); 
}

void read_matr(int *x,int m,int n)
{
  int i,j;
  for(i=0;i<m;i++)
  {
    for(j=0;j<n;j++)
      scanf("%d",&MELT(x,i,j,m,n));
    //scanf("\n");
  }
}

void read_vect(int *x,int m)
{
  int i;
  for(i=0;i<m;i++)
  {
    scanf("%d",x+i);
  }
}

void cpy_matr(int *x,int m,int n,int*x1)
{
  memcpy(x1,x,MATRIX_SIZE(m,n,int));
}

void print_matr(int *x,int m,int n)
{
  int i,j;
  for(i=0;i<m;i++)
  {
    for(j=0;j<n;j++)
      printf("%10d ",MELT(x,i,j,m,n));
    printf("\n");
  }
}

void print_vect(int *x,int m)
{
  int i;
  for(i=0;i<m;i++)
  {
    printf("%d ",x[i]);
  }
  printf("\n");
}


__global__ void fire_arc(int *bsp, int *bsw, int *mu, int mm, int n, int *y, int dbg) // grid=n, block=mm
{
  int ps = threadIdx.x;
  int t = blockIdx.x;
  MELT(y,ps,t,mm,n) = (MELT(bsw,ps,t,mm,n)>0)? mu[MELT(bsp,ps,t,mm,n)] / MELT(bsw,ps,t,mm,n) : 
                              (MELT(bsw,ps,t,mm,n)<0)? ((mu[MELT(bsp,ps,t,mm,n)]>0)? 0: INT_MAX): INT_MAX;
  //__syncthreads();
} // end of fire_arc


__global__ void fire_trs(int mm, int n, int *y, int dbg) // grid=n, block=1
{
   int t = blockIdx.x;
   int ps;
   for(ps=1;ps<mm;ps++)
     MELT(y,0,t,mm,n)=zmin(MELT(y,0,t,mm,n),MELT(y,ps,t,mm,n));
   //__syncthreads();
} // end of fire_trs


__global__ void choose_f_trs(int mm, int n, int *y, int dbg) // grid=1, block=1
{
  int t;
	 
  for(t=0; t<n; t++)
  {
    if(MELT(y,0,t,mm,n)>0)
    {
      MELT(y,0,1,mm,n)=MELT(y,0,t,mm,n); 
      MELT(y,0,0,mm,n)=t; 
      return;
    }
  }
  MELT(y,0,0,mm,n)=0; 
  MELT(y,0,1,mm,n)=0;
} // end of choose_f_trs


__global__ void next_mu(int *bsp, int *bsw, int *dsp, int *dsw, int *mu, int mm, int n, int tf, int cf, int dbg) // grid=1, block=mm
{
    int ps = threadIdx.x; 
 
    if(MELT(bsw,ps,tf,mm,n)>0) mu[MELT(bsp,ps,tf,mm,n)]-=cf*MELT(bsw,ps,tf,mm,n);
    if(MELT(dsw,ps,tf,mm,n)>0) mu[MELT(dsp,ps,tf,mm,n)]+=cf*MELT(dsw,ps,tf,mm,n);
    //__syncthreads();
} // end of next_mu    


int main(int argc, char * argv[])
{
  int m, n, mm;
  int *bsp, *bsw, *dsp, *dsw, *mu;
  int *d_bsp, *d_bsw, *d_dsp, *d_dsw, *d_mu, *d_y;
  int k=0, dbg=0, maxk=-1;
  int f[2];
  
  double t1, dt;
  
  if(argc>1) dbg=atoi(argv[1]);
  if(argc>2) maxk=atoi(argv[2]);
  
  // read sns
  
    scanf("%d %d %d\n", &m, &n, &mm);
if(dbg>0)printf("%d %d %d\n", m, n, mm);
  
  bsp=(int *)malloc(MATRIX_SIZE(mm,n,int));
  bsw=(int *)malloc(MATRIX_SIZE(mm,n,int));
  dsw=(int *)malloc(MATRIX_SIZE(mm,n,int));
  dsp=(int *)malloc(MATRIX_SIZE(mm,n,int));
  mu=(int *)malloc(VECTOR_SIZE(m,int));
  if( bsp==NULL || dsp==NULL || bsw==NULL || dsw==NULL || mu==NULL )
  {
    printf("*** error: not enough memory\n");
    exit(3);
  }
  
  read_matr(bsp,mm,n);
if(dbg>2){
printf("bsp:\n");
print_matr(bsp,mm,n);}
  read_matr(bsw,mm,n);
if(dbg>2){
printf("bsw:\n");
print_matr(bsw,mm,n);}

  read_matr(dsp,mm,n);
if(dbg>2){
printf("dsp:\n");
print_matr(dsp,mm,n);}
read_matr(dsw,mm,n);
if(dbg>2){
printf("dsw:\n");
print_matr(dsw,mm,n);}

  read_vect(mu,m);
printf("initial mu:\n");
print_vect(mu,m);
 
  // allocate device memory & copy to device
   
  CHECK(cudaSetDevice(0));
  CHECK(cudaMalloc((int**)&d_bsp, MATRIX_SIZE(mm,n, int)));
  CHECK(cudaMalloc((int**)&d_bsw, MATRIX_SIZE(mm,n, int)));
  CHECK(cudaMalloc((int**)&d_dsp, MATRIX_SIZE(mm,n, int)));
  CHECK(cudaMalloc((int**)&d_dsw, MATRIX_SIZE(mm,n, int)));
  CHECK(cudaMalloc((int**)&d_mu, VECTOR_SIZE(m,int)));
  CHECK(cudaMalloc((int**)&d_y, MATRIX_SIZE(mm,n, int)));
  
  CHECK(cudaMemcpy(d_bsp, bsp, MATRIX_SIZE(mm,n, int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_bsw, bsw, MATRIX_SIZE(mm,n, int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_dsp, dsp, MATRIX_SIZE(mm,n, int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_dsw, dsw, MATRIX_SIZE(mm,n, int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_mu, mu, VECTOR_SIZE(m,int), cudaMemcpyHostToDevice));
  
  // define device grid & block
    
  dim3 block (mm);
  dim3 grid  (n);
  dim3 block1 (1);
  dim3 grid1  (n);
  dim3 block2 (1);
  dim3 grid2  (1);
  dim3 block3 (mm);
  dim3 grid3  (1);
  
  // loop of firing transitions
  
  t1=seconds();
  
    while(maxk==-1 || k<maxk)
    {
      if(dbg>0)printf("sn-vm step k=%d\n",k); 
      
      // d_y[ps][t] = arc firing multiplicity
      fire_arc<<<grid, block>>>(d_bsp, d_bsw, d_mu, mm, n, d_y, dbg);
      CHECK(cudaDeviceSynchronize());
if(dbg>1){
CHECK(cudaMemcpy(bsp, d_y, MATRIX_SIZE(mm,n, int), cudaMemcpyDeviceToHost));
printf("y#1:\n");                              
print_matr(bsp,mm,n);} 
            
      // d_y[0][j] = reduce_min_{ps}( d_y[ps][t] );  transition firing multiplicity
      fire_trs<<<grid1, block1>>>(mm, n, d_y, dbg);
      CHECK(cudaDeviceSynchronize());          
if(dbg>1){
CHECK(cudaMemcpy(bsp, d_y, MATRIX_SIZE(mm,n, int), cudaMemcpyDeviceToHost));
printf("y#2:\n");                              
print_matr(bsp,mm,n);} 
      
      // choose firing transition (the first firable, since transitions are pre-sorted on priority)
      // d_y[0][0] = firing trs number; d_y[0][1] = firing trs multiplicity.
      choose_f_trs<<<grid2, block2>>>(mm, n, d_y, dbg);
      CHECK(cudaDeviceSynchronize());
      CHECK(cudaMemcpy(f, d_y, VECTOR_SIZE(2,int), cudaMemcpyDeviceToHost));
if(dbg>1){
CHECK(cudaMemcpy(bsp, d_y, MATRIX_SIZE(mm,n, int), cudaMemcpyDeviceToHost));
printf("y#3:\n");                              
print_matr(bsp,mm,n);} 
      
      if(f[1])
      {
        if(dbg>0)printf("fire %d in %d copies\n",f[0],f[1]);      

        // next marking
        next_mu<<<grid3, block3>>>(d_bsp, d_bsw, d_dsp, d_dsw, d_mu, mm, n, f[0], f[1], dbg);
        CHECK(cudaDeviceSynchronize()); 
        k++;
      }
      else break;
if(dbg>0){
CHECK(cudaMemcpy(mu, d_mu, VECTOR_SIZE(m,int), cudaMemcpyDeviceToHost));
printf("mu:\n");                              
print_vect(mu,m);} 

// *** 
if(dbg>2){
CHECK(cudaMemcpy(bsp, d_bsp, MATRIX_SIZE(mm,n, int), cudaMemcpyDeviceToHost));
printf("bsp:\n");
print_matr(bsp,mm,n);
CHECK(cudaMemcpy(bsw, d_bsw, MATRIX_SIZE(mm,n, int), cudaMemcpyDeviceToHost));
printf("bsw:\n");
print_matr(bsw,mm,n);
CHECK(cudaMemcpy(dsp, d_dsp, MATRIX_SIZE(mm,n, int), cudaMemcpyDeviceToHost));
printf("dsp:\n");
print_matr(dsp,mm,n);
CHECK(cudaMemcpy(dsw, d_dsw, MATRIX_SIZE(mm,n, int), cudaMemcpyDeviceToHost));
printf("dsw:\n");
print_matr(dsw,mm,n);}
// ***

  } // end of while (firing transitions)
   
  dt=seconds()-t1;
  printf("it took %f s.\n",dt);
  CHECK(cudaGetLastError());
  
  // copy from device and print resulting marking
      
  CHECK(cudaMemcpy(mu, d_mu, VECTOR_SIZE(m,int), cudaMemcpyDeviceToHost));
  
  printf("final mu:\n");  
  print_vect(mu,m);
  
  // free memory of device and host
  
  CHECK(cudaFree(d_bsp));
  CHECK(cudaFree(d_bsw));
  CHECK(cudaFree(d_dsp));
  CHECK(cudaFree(d_dsw));
  CHECK(cudaFree(d_mu));
  CHECK(cudaFree(d_y));
  
  free(bsp);
  free(bsw);
  free(dsp); 
  free(dsw); 

} // end of main

// @ 2024 Tatiana R. Shmeleva: ta.arta@gmail.com

