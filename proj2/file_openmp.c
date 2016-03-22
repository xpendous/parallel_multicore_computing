/* C code for Simulating Annealing on TSP -- David Bookstaber */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define N 1024 // Number of cities
#define K 70 // Dimension of state graph (range of interchange)
#define ITS 1000L // Iterations
#define PMELT 0.7 // Fraction of melting point for starting temperature
#define TARGT 0.01 // Fraction of melting point for ending temperature
#define STAGFACTOR 0.05 // Fraction of runs to allow stagnant before reheating
typedef struct {int list[N]; float fit;} state;
float city[N][2];

double startime,endtime;
int i, j, k, n;
long l, optgen;
double minf, maxf, range, Dtemp;
state x, optimum;


int max ( int a, int b ) { return a > b ? a : b; }

float distance(int a, int b) {
	return(pow(pow(city[a][0]-city[b][0],2)+pow(city[a][1]-city[b][1],2),0.5));
}

float fitness(state x) {
	int i;
	float sum = distance(x.list[0],x.list[N-1]);
	#pragma omp parallel for reduction(+:sum) lastprivate(i)
	for(i = 1;i < N;i++) 
		sum += distance(x.list[i],x.list[i-1]);
	return(sum);
}

state iterate(float temp,state x) {
	int i, pt, sh;
	//state y = x;
	state y;
	y.fit = x.fit;
	
	#pragma omp parallel for
	for(i=0;i<N;i++){
		y.list[i] = x.list[i];	
	}
	
	pt = rand() % N;
	sh = (pt+(rand() % K)+1) % N;
	y.list[pt] = y.list[pt]^y.list[sh];
	y.list[sh] = y.list[sh]^y.list[pt];
	y.list[pt] = y.list[pt]^y.list[sh];
	y.fit = fitness(y);
	if(y.fit < x.fit) {
		return(y);
	}else if((float)rand()/(1.0*RAND_MAX) < exp(-1.0*(y.fit-x.fit)/temp))
		return(y);
	else
		return(x);
}

void init(){
	/* Initialization of city grid and state list */
	for(i = 0,k = 0,n = sqrt(N);i < n;i++) {
		for(j = 0;j < n;j++,k=n*i+j) {
			city[k][0] = i; city[k][1] = j;
			x.list[k] = k;
		}
	}
}

void randomState(){
	/* Randomization of state list--requires N Log[N] "shuffles" */
	for(i = 0,k = rand()%(N-1)+1;i < N*log(N);i++,k = rand()%(N-1)+1) {
		x.list[0] = x.list[0]^x.list[k];
		x.list[k] = x.list[k]^x.list[0];
		x.list[0] = x.list[0]^x.list[k];
	}
}


void tempeartureSchedule(){
	/* Sample state space with 1% of runs to determine temperature schedule */
	for(i = 0,maxf = 0,minf = pow(10,10),x.fit=fitness(x);i < max(0.01*N,2);i++) {
		x = iterate(pow(10,10),x);
		minf = (x.fit < minf) ? x.fit : minf;
		maxf = (x.fit > maxf) ? x.fit : maxf;
	}
	range = (maxf - minf)*PMELT;
	Dtemp = pow(TARGT,1.0/ITS);
}
void main() {
	omp_set_num_threads(1);
	//srand((int)time(NULL));
	srand(1);
	init();
	randomState();
	tempeartureSchedule();

	startime = omp_get_wtime();
	
	/* Simulate Annealing */
	for(optgen = l = 1,optimum.fit = x.fit;l < ITS;l++) {
		
		{	
			x = iterate(range*pow(Dtemp,l),x);
			if(x.fit < optimum.fit) {
				
				optimum.fit = x.fit;
				#pragma omp parallel for
				for(i=0;i<N;i++){
					optimum.list[i] = x.list[i];	
				}
				optgen = l;
			}
			/* Reheat if stagnant */
			if(l-optgen == STAGFACTOR*ITS) 
				Dtemp = pow(Dtemp,STAGFACTOR*l/ITS);
		}			
		/* Graphics */
		printf("Iteration: %ld\n",l);
		printf("Fitness: %f\t\tTemp: %f\t\n",x.fit,range*pow(Dtemp,l));
		//printf("Current Optimum %f found on %ld\t\n",optimum.fit,optgen);
		printf("Global Optimum %f found on %ld\t\n",optimum.fit,optgen);
		//printf("Global Optimum is %d\n\n",N);
		//printf("Sample Range: %f\tTemp decrement: %f\n\n",range,Dtemp);
		
		/* End Graphics */
	}

	endtime = omp_get_wtime();
	printf("elapsed time: %f\n", endtime-startime);


}
