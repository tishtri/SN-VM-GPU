// Convertor of MSN to MCC
//
// MSN: Sleptsov Net Raw Matrix File Format
// MCC: Sleptsov Net Matrix with Condensed Columns Format
//
// Compile: gcc -o msn-to-mcc msn-to-mcc.c
//
// Run: ./msn-to-mcc < msn_file > mcc_file

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define MATRIX_SIZE(d1,d2,t) ((d1)*(d2)*(sizeof(t)))
#define VECTOR_SIZE(d1,t)    ((d1)*(sizeof(t)))

#define MOFF(i,j,d1,d2) ((d2)*(i)+(j))
#define MELT(x,i,j,d1,d2) (*((x)+MOFF(i,j,d1,d2)))

#define zmax(x,y) (((x)>(y))?(x):(y))
#define zmin(x,y) (((x)<(y))?(x):(y))

void zero_matr(int *x,int m,int n)
{
  memset(x,0,MATRIX_SIZE(m,n,int));
}

void zero_mu(int *x,int m)
{
  memset(x,0,VECTOR_SIZE(m,int)); 
}

int read_matr(int *x,int m,int n)
{
  int i,j;
  for(i=0;i<m;i++)
  {
    for(j=0;j<n;j++)
      scanf("%d",&MELT(x,i,j,m,n));
    //scanf("\n");
  }
}

int read_vect(int *x,int m)
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

int print_mu(int *mu,int m)
{
  int i,p,v;
  for(i=0;i<m;i++)
  {
    v=mu[i];
    if(v) printf("%d %d\n",i+1,v);
  }
}

// maximal number of nonzero elements over columns

int max_nz_col(int *x, int m, int n)
{
   int p, t, mmc, mm=0;
   
   for(t=0; t<n; t++)
   {
     mmc=0;
     for(p=0; p<m; p++)
     {
       if( MELT(x,p,t,m,n) != 0 ) mmc++;
     }
     mm=zmax(mm,mmc);
   }
   
   return( mm );
} // end of max_nz_col


int belong_to(int *v, int next, int x) // x belongs to v
{
  int t;
  for(t=0; t<next; t++)
    if( v[t] == x ) return (1);
  return(0);
} // end of belong_to


// compose permutation to sort transitions by priorities

void find_t_perm(int *r, int n, int *t_perm)  
{
  int t,tt,tr,tc=0;
  int ng,ts,tf;
  
  // find maximal transitions
  
  for(tt=0; tt<n; tt++)
  {
    ng=0;
    for(t=0; t<n; t++)
    {
      if( MELT(r,t,tt,n,n) != 0 ) ng++;
    }
    if( ng==0 )
    {
      t_perm[tc++]=tt;
    }
  }
  
  ts=0;
  tf=tc;
  
  while( tc < n )
  {
    for(tr=ts; tr<tf; tr++)
    {
      t=t_perm[tr];
      for(tt=0; tt<n; tt++)
      {
        if( MELT(r,t,tt,n,n) != 0 )
        {
          if( !belong_to(t_perm,tc,tt) )
            t_perm[tc++]=tt;
        }
      }
    }
    
    ts=tf;
    tf=tc;
  }
  
} // end of find_t_perm

// build sparse matrices of a raw matrix

void build_sparse_matr(int *x, int m, int n, int *t_perm, int mm, int *xsp, int*xsw)
{
   int t,tt,p,ps;

   zero_matr(xsp,mm,n);
   zero_matr(xsw,mm,n);
   
   for(t=0; t<n; t++)
   {
     tt=t_perm[t];
     ps=0;
     for(p=0; p<m; p++)
     {
       if( MELT(x,p,tt,m,n) !=0 ) 
       {
         MELT(xsp,ps,t,mm,n) = p;
         MELT(xsw,ps,t,mm,n) = MELT(x,p,tt,m,n);
         ps++;
       }
     }
   }

} // end of build_sparse_matr


int main(int argc, char * argv[])
{
  int m, n, dbg=0;
  int *b, *d, *r, *mu;
  
  if(argc>1)dbg=atoi(argv[1]);
  
  // read raw matrix SN
  
  scanf("%d %d\n", &m, &n);
if(dbg>1)printf("%d %d\n", m, n);
  
  b=malloc(MATRIX_SIZE(m,n,int));
  d=malloc(MATRIX_SIZE(m,n,int));
  r=malloc(MATRIX_SIZE(n,n,int));
  mu=malloc(VECTOR_SIZE(m,int));
  
  read_matr(b,m,n);
if(dbg>1){
printf("b:\n");
print_matr(b,m,n);}
  read_matr(d,m,n);
if(dbg>1){
printf("d:\n");
print_matr(d,m,n);}
  read_matr(r,n,n);
if(dbg>1){
printf("r:\n");
print_matr(r,n,n);}
  read_vect(mu,m);
if(dbg>1){printf("mu:\n");
print_vect(mu,m);}

// convert to sparse matrix

  int mm=0;
  int *bsp, *bsw, *dsp, *dsw, *t_perm;

  mm=zmax(mm,max_nz_col(b,m,n));
  mm=zmax(mm,max_nz_col(d,m,n));

  bsp=malloc(MATRIX_SIZE(mm,n,int));
  dsp=malloc(MATRIX_SIZE(mm,n,int));
  bsw=malloc(MATRIX_SIZE(mm,n,int));
  dsw=malloc(MATRIX_SIZE(mm,n,int));
  t_perm=malloc(VECTOR_SIZE(n,int));
  
  find_t_perm(r,n,t_perm);
  
//printf("\n# permutation of transitions:\n");
//print_vect(t_perm,n); 
  
  build_sparse_matr(b,m,n,t_perm,mm,bsp,bsw);
  build_sparse_matr(d,m,n,t_perm,mm,dsp,dsw);
 
  free(r);
  free(d);
  free(b); 
  
// output sparse matrix SN

  printf("%d %d %d\n", m, n, mm);
  print_matr(bsp,mm,n); printf("\n");
  print_matr(bsw,mm,n); printf("\n");
  print_matr(dsp,mm,n); printf("\n");
  print_matr(dsw,mm,n); printf("\n");
  print_vect(mu,m); 
  printf("\n# permutation of transitions:\n");
  print_vect(t_perm,n); 
  
// free memory  

  free(bsp);
  free(bsw);
  free(dsp); 
  free(dsw);
  free(t_perm);
  free(mu); 

}

//  2024 Tatiana R. Shmeleva: ta.arta@gmail.com
