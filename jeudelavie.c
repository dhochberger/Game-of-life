// Jeu de la vie avec sauvegarde de quelques itérations
// compiler avec gcc -O3 -march=native (et -fopenmp si OpenMP souhaité)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

// hauteur et largeur de la matrice
#define HM 1200
#define LM 800

// nombre total d'itérations
#define ITER 10001
// multiple d'itérations à sauvegarder
#define SAUV 1000

#define DIFFTEMPS(a,b) \
(((b).tv_sec - (a).tv_sec) + ((b).tv_usec - (a).tv_usec)/1000000.)

/* tableau de cellules */
typedef char Tab[HM][LM];

// initialisation du tableau de cellules
void init(Tab);

// calcule une nouveau tableau de cellules à partir de l'ancien
// - paramètres : ancien, nouveau
void calcnouv(Tab, Tab);

// variables globales : pas de débordement de pile
Tab t1, t2;
Tab tsauvegarde[1+ITER/SAUV];
int rank,size;
MPI_Request requestRecv, requestSend, requestRecvBis, requestSendBis;
MPI_Status status, statusBis;



int main( int argc, char **argv )
{
  struct timeval tv_init, tv_beg, tv_end, tv_save;

  MPI_Init(&argc,&argv);
  int root;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  gettimeofday( &tv_init, NULL);

  if (rank == 0)
    init(t1);


  MPI_Scatter(t1, HM*LM/size, MPI_CHAR, t1[HM/size*rank], HM*LM/size, MPI_CHAR, 0, MPI_COMM_WORLD);

  gettimeofday( &tv_beg, NULL);


  for(int i=0 ; i<ITER ; i++)
  {

    if( i%2 == 0)
    {
      calcnouv(t1, t2);
    }
    else
    {
      calcnouv(t2, t1);
    }

    if(i%SAUV == 0)
    {
      // Utile pour les sauvegardes
      MPI_Gather(t1[HM/size*rank], HM*LM/size, MPI_CHAR, t1, HM*LM/size, MPI_CHAR, 0, MPI_COMM_WORLD);
      if (rank==0)
      {

        printf("sauvegarde (%d)\n", i);
        // copie t1 dans tsauvegarde[i/SAUV]
        for(int x=0 ; x<HM ; x++)
          for(int y=0 ; y<LM ; y++)
            tsauvegarde[i/SAUV][x][y] = t1[x][y];
      }
    }
  }

  gettimeofday( &tv_end, NULL);
  if (rank==0)
  {
    FILE *f = fopen("jdlv.out", "w");
    for(int i=0 ; i<ITER ; i+=SAUV)
    {
      fprintf(f, "------------------ sauvegarde %d ------------------\n", i);
      for(int x=0 ; x<HM ; x++)
      {
        for(int y=0 ; y<LM ; y++)
          fprintf(f, tsauvegarde[i/SAUV][x][y]?"*":" ");
        fprintf(f, "\n");
      }
    }
  fclose(f);
  }
  gettimeofday( &tv_save, NULL);

  printf("init : %lf s,", DIFFTEMPS(tv_init, tv_beg));
  printf(" calcul : %lf s,", DIFFTEMPS(tv_beg, tv_end));
  printf(" sauvegarde : %lf s\n", DIFFTEMPS(tv_end, tv_save));

  MPI_Finalize();


  return( 0 );
}

void init(Tab t)
{
  srand(time(0));
  for(int i=0 ; i<HM ; i++)
    for(int j=0 ; j<LM ; j++ )
    {
      // t[i][j] = rand()%2;
      // t[i][j] = ((i+j)%3==0)?1:0;
      // t[i][j] = (i==0||j==0||i==h-1||j==l-1)?0:1;
      t[i][j] = 0;
    }
  t[10][10] = 1;
  t[10][11] = 1;
  t[10][12] = 1;
  t[9][12] = 1;
  t[8][11] = 1;

  t[55][50] = 1;
  t[54][51] = 1;
  t[54][52] = 1;
  t[55][53] = 1;
  t[56][50] = 1;
  t[56][51] = 1;
  t[56][52] = 1;
}

int nbvois(Tab t, int i, int j)
{
  int n=0;
  if( i>0 )
  {  /* i-1 */
    if( j>0 )
      if( t[i-1][j-1] )
        n++;
    if( t[i-1][j] )
        n++;
    if( j<LM-1 )
      if( t[i-1][j+1] )
        n++;
  }
  if( j>0 )
    if( t[i][j-1] )
      n++;
  if( j<LM-1 )
    if( t[i][j+1] )
      n++;
  if( i<HM-1 )
  {  /* i+1 */
    if( j>0 )
      if( t[i+1][j-1] )
        n++;
    if( t[i+1][j] )
        n++;
    if( j<LM-1 )
      if( t[i+1][j+1] )
        n++;
  }
  return( n );
}

void calcnouv(Tab t, Tab n)	{

	if (size > 1)
	{
	  if (rank != size-1 )
	  {
  		MPI_Issend(t[(HM/size)*(rank+1)-1], LM, MPI_CHAR, rank+1, 0, MPI_COMM_WORLD, &requestSend);
  		MPI_Irecv(t[(HM/size)*(rank+1)], LM, MPI_CHAR, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, &requestRecv);
	  }
	  if (rank != 0)
	{
  		MPI_Issend(t[(HM/size)*rank], LM, MPI_CHAR, rank-1, 0, MPI_COMM_WORLD, &requestSendBis);
  		MPI_Irecv(t[(HM/size)*rank-1], LM, MPI_CHAR, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &requestRecvBis);
	  }
  }

  #pragma omp parallel for
  for(int i=(HM/size)*rank; i<((HM/size)*(rank+1)); i++)
  {

   if (size>1)
   {
	 if (rank != (size-1) && i == (HM/size)*(rank+1)-1)
	 {
        MPI_Wait(&requestRecv,&status);
      }
      else if (rank != 0 && i == ((HM/size)*rank)){
        i++;
      }
    }

    for(int j=0 ; j<LM ; j++)
    {
      int v = nbvois(t, i, j);
      if(v==3)
        n[i][j] = 1;
      else if(v==2)
        n[i][j] = t[i][j];
      else
        n[i][j] = 0;
    }
  }

  if (rank != 0)
  {
    MPI_Wait(&requestRecvBis,&statusBis);

    for(int j=0 ; j<LM ; j++)
    {
      int v = nbvois(t, HM/size*rank, j);
      if(v==3)
        n[HM/size*rank][j] = 1;
      else if(v==2)
        n[HM/size*rank][j] = t[HM/size*rank][j];
      else
        n[HM/size*rank][j] = 0;
    }
  }

}
