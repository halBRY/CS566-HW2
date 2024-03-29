/* Gaussian elimination without pivoting.
 * Compile with "gcc gauss_pt.c -pthread" 
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */

int threadNum;
int N;  /* Matrix size */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
		* It is this routine that is timed.
		* It is called only on the parent.
		*/

void* gauss_elim_paralell();

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int seed = 0;  /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */

  if (argc == 4) {
    seed = atoi(argv[2]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  } 
  if (argc >= 2) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
    threadNum = atoi(argv[3]);
  }
  else {
    printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);    
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }

    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;

    /*
    A[0][0] = 1;
    A[0][1] = 1;
    A[0][2] = -1;

    A[1][0] = 2;
    A[1][1] = -1;
    A[1][2] = 1;

    A[2][0] = -1;
    A[2][1] = 2;
    A[2][2] = 2;

    B[0] = -2;
    B[1] = 5;
    B[2] = 1;
    */
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X() {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char **argv) {
  /* Timing variables */
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  /* Process program parameters */
  parameters(argc, argv);

  /* Initialize A and B */
  initialize_inputs();

  /* Print input matrices */
  print_inputs();

  /* Start Clock */
  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  /* Gaussian Elimination */
  gauss();

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("\nStopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_X();

  /* Display timing results */
  printf("\nElapsed time = %g ms.\n",
	 (float)(usecstop - usecstart)/(float)1000);

  printf("(CPU times are accurate to the nearest %g ms)\n",
	 1.0/(float)CLOCKS_PER_SEC * 1000.0);
  printf("My total CPU time for parent = %g ms.\n",
	 (float)( (cputstop.tms_utime + cputstop.tms_stime) -
		  (cputstart.tms_utime + cputstart.tms_stime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My system CPU time for parent = %g ms.\n",
	 (float)(cputstop.tms_stime - cputstart.tms_stime) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My total CPU time for child processes = %g ms.\n",
	 (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
		  (cputstart.tms_cutime + cputstart.tms_cstime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
      /* Contrary to the man pages, this appears not to include the parent */
  printf("--------------------------------------------\n");
  
  exit(0);
}

/* ------------------ Above Was Provided --------------------- */

struct thread_info{
  int id;
  int normNum;
  int chunkSize;
};

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
  void gauss() {
    int norm;

    printf("Computing Parallel with pThreads.\n\n");

    for (norm = 0; norm < N - 1; norm++) {

      struct thread_info threads_info[threadNum];

      pthread_t threads[threadNum];


      for(int i = 0; i < threadNum; i++)
      {
        threads_info[i].chunkSize = threadNum;
        threads_info[i].normNum = norm;
        threads_info[i].id = i;
        //printf("Creating thread %d for norm %d\n", i, norm);
        pthread_create(&threads[i], NULL, gauss_elim_paralell, (void *) &threads_info[i]);
      } 

      for(int i = 0; i < threadNum; i++)
      {
        pthread_join(threads[i], NULL);
      }

    }

  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */

  /* Back substitution */
  int row, col;
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N-1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}

void* gauss_elim_paralell(void *args)
{
  struct thread_info *my_info;
  my_info = (struct thread_info *) args;

  //printf("I am thread %d from norm %d\n", my_info->id, my_info->normNum);
  //print_inputs();

  int norm = my_info->normNum;
  int chunk = my_info->chunkSize;
  int row = my_info->id + (norm + 1); 
  int col = 0;
  float multiplier = 0;

  int rows = ceil(N/threadNum);

  if(rows == 0)
    rows = 1;

  for (row; row <= N; row += chunk) 
  {
    if(row < N)
    {
      multiplier = A[row][norm] / A[norm][norm];

      //printf("Thread %d: Working on row %d for norm %d. My next row is row %d.\n", my_info->id, row, norm, row+chunk);
      for (col = norm; col < N; col++) 
      {
        //printf("Thread %d: Working on row %d col %d for norm %d.\n", my_info->id, row, col, norm);
        A[row][col] -= A[norm][col] * multiplier;       
      }
      //printf("Thread %d: Writing %f to B row %d for norm %d with a multiplier of %f.\n", my_info->id, (B[row] - (B[norm] * multiplier)), row, norm, multiplier);
      B[row] -= B[norm] * multiplier;
    }
    else
    {
      break;
    }

  }
}
