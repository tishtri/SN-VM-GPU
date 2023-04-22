// Sleptsov net Virtual Machine on GPU
// Uses a single GPU block only with a single kernel program 
// Compatibility with NVIDIA architecture 35
// Compile: nvcc sn-vm-gpu-1b.cu -o sn-vm-gpu -gencode arch=compute_35,code=compute_35 -Wno-deprecated-gpu-targets
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
      printf("%2d ",MELT(x,i,j,m,n));
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

void print_mu(int *mu,int m)
{
  int i,v;
  for(i=0;i<m;i++)
  {
    v=mu[i];
    if(v) printf("%d %d\n",i+1,v);
  }
}

__global__ void sn_vm_gpu(int *b, int *d, int *r, int *mu, int m, int n, int *y, int dbg)
{
    int j = threadIdx.x;
    
    __shared__ int nf, jf, cf, k, go;

    int ii,jj;  
    
    k=0; go=1;
    while(go)
    {
if(j==0&&dbg>0)printf("sn-vm step j=%d, k=%d\n",j,k); 
      // y[i][j] = arc firing multiplicity
      
      if(j<n)
       for(ii=0;ii<m;ii++)
         MELT(y,ii,j,m,n) = (MELT(b,ii,j,m,n)>0)? mu[ii] / MELT(b,ii,j,m,n) : 
                              (MELT(b,ii,j,m,n)<0)? ((mu[ii]>0)? 0: INT_MAX): INT_MAX;

      
if(j==0&&dbg>1) // print y
{
  printf("y#1:\n");                              
  for(ii=0;ii<m;ii++)
  {
    for(jj=0;jj<n;jj++)
      printf("%10d ",MELT(y,ii,jj,m,n));
    printf("\n");
  }
}                           
      __syncthreads();
      // y[0][j] = red_min( y[j] );  transition firing multiplicity
     if(j<n) 
       for(ii=1;ii<m;ii++)
          MELT(y,0,j,m,n)=zmin(MELT(y,0,j,m,n),MELT(y,ii,j,m,n));
          
          
if(j==0&&dbg>1) // print y
{
  printf("y#2:\n");                              
  for(ii=0;ii<1;ii++)
  {
    for(jj=0;jj<n;jj++)
      printf("%10d ",MELT(y,ii,jj,m,n));
    printf("\n");
  }
}              
      __syncthreads();
      // remove low priority transitions
      if(j<n)
        for(ii=0;ii<n;ii++)
        {
          if(ii==j) continue;
          if(MELT(r,ii,j,n,n) > 0 && MELT(y,0,ii,m,n) != 0) 
            MELT(y,0,j,m,n)=0;
        }
        
              
if(j==0&&dbg>1) // print y
{
  printf("y#3:\n");                              
  for(ii=0;ii<1;ii++)
  {
    for(jj=0;jj<n;jj++)
      printf("%10d ",MELT(y,ii,jj,m,n));
    printf("\n");
  }
}                 
      __syncthreads();
      // count firable and choose firing
      if(j==0)
      {
        nf=0;	 
        for(jj=0;jj<n;jj++)
	  nf+=(MELT(y,0,jj,m,n)>0) ? 1 : 0; 

        if(nf==0) go=0;
      if(go){
        for(jj=0;jj<n;jj++)  
          if(MELT(y,0,jj,m,n)>0)
	  {
	    if(nf==1) {jf=jj; break;} else nf--;
	  } 
      
        cf=MELT(y,0,jf,m,n);}
      }
__syncthreads();
if(go){
if(j==0&&dbg>0)printf("fire %d in %d copies\n",jf,cf);      

       
      //printf("threads: i=%d when j=%d\n",i,j); 
      // next marking
      if(j<m)
      {
        //printf("compute mu[%d] when t%d fires in %d copies\n",i,jf,cf);
        mu[j]+=((MELT(b,j,jf,m,n)>0)?-cf*MELT(b,j,jf,m,n):0)+cf*MELT(d,j,jf,m,n);
      }
       
      __syncthreads();
 
if(j==0&&dbg>0) // print mu
{
  printf("mu:\n"); 
  for(ii=0;ii<m;ii++)
  {
    printf("%d ",mu[ii]);
  }
  printf("\n");
}

      k++;   }
    } 
    
  
}


int main(int argc, char * argv[])
{
  int m, n;
  int *b, *d, *r, *mu;
  int *d_b, *d_d, *d_r, *d_mu, *d_y;
  int mn, dbg=0;
  
  double t1, dt;
  
  if(argc>1) dbg=atoi(argv[1]);
  
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
  
  mn=zmax(m,n);
 
  dim3 block (mn);
  dim3 grid  (1);
  
  t1=seconds();
  sn_vm_gpu<<<grid, block>>>(d_b, d_d, d_r, d_mu, m, n, d_y, dbg);
  CHECK(cudaDeviceSynchronize());
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

