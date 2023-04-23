// Sleptsov net Virtual Machine on GPU
// Uses variable number of GPU blocks with a few kernel programs
// Compatibility with NVIDIA architecture 35
// Compile: nvcc sn-vm-gpu-fk.cu -o sn-vm-gpu-fk -gencode arch=compute_35,code=compute_35 -Wno-deprecated-gpu-targets
// No GPU timeout: sudo systemctl isolate multi-user.target
// Run:     ./sn-vm-gpu < net.mat
// @ 2023 Tatiana R. Shmeleva: tatianar.shmeleva@gmail.com

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

__global__ void fire_arc(int *b, int *mu, int m, int n, int *y, int dbg)
{
  int i = threadIdx.x;
  int j = blockIdx.x;
  MELT(y,i,j,m,n) = (MELT(b,i,j,m,n)>0)? mu[i] / MELT(b,i,j,m,n) : 
                              (MELT(b,i,j,m,n)<0)? ((mu[i]>0)? 0: INT_MAX): INT_MAX;
  //__syncthreads();
}

__global__ void fire_trs(int m, int n, int *y, int dbg)
{
   int j = threadIdx.x;
   int ii;
   for(ii=1;ii<m;ii++)
          MELT(y,0,j,m,n)=zmin(MELT(y,0,j,m,n),MELT(y,ii,j,m,n));
}

__global__ void rem_l_pri(int *r, int m, int n, int *y, int dbg)
{
  int j = threadIdx.x;
  int i = blockIdx.x;
  if(i!=j)
  {
    if(MELT(r,i,j,n,n) > 0 && MELT(y,0,i,m,n) != 0) 
      MELT(y,0,j,m,n)=0;
  }
  
}

__global__ void choose_f_trs(int m, int n, int *y, int dbg)
{
  int jj;
  __shared__ int nf;

  nf=0;	 
  for(jj=0;jj<n;jj++)
    nf+=((MELT(y,0,jj,m,n)>0) ? 1 : 0); 

  if(nf==0) 
  {
    MELT(y,0,0,m,n)=0; 
    MELT(y,0,1,m,n)=0;
  }
  else
  {
    //nf=rand()%nf+1;
    nf=1;
    for(jj=0;jj<n;jj++)  
    {
      if(MELT(y,0,jj,m,n)>0)
      {
        nf--;
	if(nf==0)
	{
	  MELT(y,0,1,m,n)=MELT(y,0,jj,m,n); 
	  MELT(y,0,0,m,n)=jj; 
	  break;
	}
      } 
    }
    
  }
  
}

__global__ void next_mu(int *b, int *d, int *mu, int m, int n, int *y, int jf, int cf, int dbg)
{
    int i = threadIdx.x;
    mu[i]+=((MELT(b,i,jf,m,n)>0)?-cf*MELT(b,i,jf,m,n):0)+cf*MELT(d,i,jf,m,n);
    //__syncthreads();
}        


int main(int argc, char * argv[])
{
  int m, n;
  int *b, *d, *r, *mu;
  int *d_b, *d_d, *d_r, *d_mu, *d_y;
  int k=0, dbg=0, maxk=-1;
  int f[2];
  
  double t1, dt;
  
  if(argc>1) dbg=atoi(argv[1]);
  if(argc>2) maxk=atoi(argv[2]);
  
  scanf("%d %d\n", &m, &n);
if(dbg>0)printf("%d %d\n", m, n);
  
  b=(int *)malloc(MATRIX_SIZE(m,n,int));
  d=(int *)malloc(MATRIX_SIZE(m,n,int));
  r=(int *)malloc(MATRIX_SIZE(n,n,int));
  mu=(int *)malloc(VECTOR_SIZE(m,int));
  if( b==NULL || d==NULL || r==NULL || mu==NULL )
  {
    printf("*** error: not enough memory\n");
    exit(3);
  }
  
  read_matr(b,m,n);
if(dbg>2){
printf("b:\n");
print_matr(b,m,n);}
  read_matr(d,m,n);
if(dbg>2){
printf("d:\n");
print_matr(d,m,n);}
  read_matr(r,n,n);
if(dbg>2){
printf("r:\n");
print_matr(r,n,n);}
  read_vect(mu,m);
printf("initial mu:\n");
print_vect(mu,m);

  srandom(m+n+((unsigned long)b%INT_MAX));
 
  //y=malloc(MATRIX_SIZE(m,n, int));
  
  CHECK(cudaSetDevice(0));
  CHECK(cudaMalloc((int**)&d_b, MATRIX_SIZE(m,n, int)));
  CHECK(cudaMalloc((int**)&d_d, MATRIX_SIZE(m,n, int)));
  CHECK(cudaMalloc((int**)&d_r, MATRIX_SIZE(n,n, int)));
  CHECK(cudaMalloc((int**)&d_mu, VECTOR_SIZE(m,int)));
  
  CHECK(cudaMalloc((int**)&d_y, MATRIX_SIZE(m,n, int)));
  
  CHECK(cudaMemcpy(d_b, b, MATRIX_SIZE(m,n, int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_d, d, MATRIX_SIZE(m,n, int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_r, r, MATRIX_SIZE(n,n, int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_mu, mu, VECTOR_SIZE(m,int), cudaMemcpyHostToDevice));
  
  dim3 block (m);
  dim3 grid  (n);
  dim3 block1 (n);
  dim3 grid1  (1);
  dim3 block2 (1);
  dim3 grid2  (1);
  dim3 block3 (m);
  dim3 grid3  (1);
  dim3 block4 (n);
  dim3 grid4  (n);
  
  t1=seconds();
  //sn_vm_gpu<<<grid, block>>>(d_b, d_d, d_r, d_mu, m, n, d_y, dbg);
    while(maxk==-1 || k<maxk)
    {
      if(dbg>0)printf("sn-vm step k=%d\n",k); 
      // y[i][j] = arc firing multiplicity
      
      fire_arc<<<grid, block>>>(d_b, d_mu, m, n, d_y, dbg);
      CHECK(cudaDeviceSynchronize());
if(dbg>1){
CHECK(cudaMemcpy(b, d_y, MATRIX_SIZE(m,n, int), cudaMemcpyDeviceToHost));
printf("y#1:\n");                              
print_matr(b,m,n);} 
            
      // y[0][j] = red_min( y[j] );  transition firing multiplicity
      fire_trs<<<grid1, block1>>>(m, n, d_y, dbg);
      CHECK(cudaDeviceSynchronize());
      
      rem_l_pri<<<grid4, block4>>>(d_r, m, n, d_y, dbg);
      CHECK(cudaDeviceSynchronize());
      
      
if(dbg>1){
CHECK(cudaMemcpy(b, d_y, MATRIX_SIZE(m,n, int), cudaMemcpyDeviceToHost));
printf("y#2:\n");                              
print_matr(b,m,n);} 
      
      // count firable and choose firing
      choose_f_trs<<<grid2, block2>>>(m, n, d_y, dbg);
      CHECK(cudaDeviceSynchronize());
      CHECK(cudaMemcpy(f, d_y, VECTOR_SIZE(2,int), cudaMemcpyDeviceToHost));
if(dbg>1){
CHECK(cudaMemcpy(b, d_y, MATRIX_SIZE(m,n, int), cudaMemcpyDeviceToHost));
printf("y#3:\n");                              
print_matr(b,m,n);} 
      
      if(f[1])
      {
        if(dbg>0)printf("fire %d in %d copies\n",f[0],f[1]);      

        // next marking
        next_mu<<<grid3, block3>>>(d_b, d_d, d_mu, m, n, d_y, f[0], f[1], dbg);
        CHECK(cudaDeviceSynchronize()); 
        k++;
      }
      else break;
if(dbg>0){
CHECK(cudaMemcpy(mu, d_mu, VECTOR_SIZE(m,int), cudaMemcpyDeviceToHost));
printf("mu:\n");                              
print_vect(mu,m);} 
  } 
   
  dt=seconds()-t1;
  printf("it took %f s.\n",dt);
  CHECK(cudaGetLastError());
      
  CHECK(cudaMemcpy(mu, d_mu, VECTOR_SIZE(m,int), cudaMemcpyDeviceToHost));
  
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_d));
  CHECK(cudaFree(d_r));
  CHECK(cudaFree(d_mu));

  printf("final mu:\n");  
  print_vect(mu,m);
    
  free(mu);
  free(r);
  free(d);
  free(b); 
}

// @ 2023 Tatiana R. Shmeleva: tatianar.shmeleva@gmail.com

