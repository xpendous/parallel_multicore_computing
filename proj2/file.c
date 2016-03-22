/* C code for Simulating Annealing on TSP -- David Bookstaber */
// gcc file.c -lm


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 100 // Number of cities
#define K 70 // Dimension of state graph (range of interchange)
#define ITS 100L // Iterations
#define PMELT 0.7 // Fraction of melting point for starting temperature
#define TARGT 0.01 // Fraction of melting point for ending temperature
#define STAGFACTOR 0.05 // Fraction of runs to allow stagnant before reheating

// list[N]: an order of cities; 
// fit: the whole distance of that order
typedef struct {int list[N]; float fit;} state; 

// coordinates of N cities
float city[N][2];

int max ( int a, int b ) { return a > b ? a : b; }

float distance(int a, int b) {
	return(pow(pow(city[a][0]-city[b][0],2)+pow(city[a][1]-city[b][1],2),0.5));
}

// given an order of cities, return the whole distance
float fitness(state x) {
	int i;
	float sum = distance(x.list[0],x.list[N-1]);
	for(i = 1;i < N;i++) sum += distance(x.list[i],x.list[i-1]);
	return(sum);
}

state iterate(float temp,state x) {
	int i, pt, sh;
	state y = x;
	pt = rand() % N;
	sh = (pt+(rand() % K)+1) % N;
	y.list[pt] = y.list[pt]^y.list[sh];
	y.list[sh] = y.list[sh]^y.list[pt];
	y.list[pt] = y.list[pt]^y.list[sh];
	y.fit = fitness(y);
	if(y.fit < x.fit) {
		return(y);
	}else if((float)rand()/1.0 < exp(-1.0*(y.fit-x.fit)/temp))
		return(y);
	else
		return(x);
}
void main() {
	int i, j, k, n;
	long l, optgen;
	double minf, maxf, range, Dtemp;
	state x, optimum;
	FILE *fp;
	//clrscr();
	srand(1);
	/* Initialization of city grid and state list */
	for(i = 0,k = 0,n = sqrt(N);i < n;i++) {
		for(j = 0;j < n;j++,k=n*i+j) {
			city[k][0] = i; city[k][1] = j;
			x.list[k] = k;
		}
	}// x.list: city order: 0,1,2,...,99



	/* Randomization of state list--requires N Log[N] "shuffles" */
	for(i = 0,k = rand()%(N-1)+1;i < N*log(N);i++,k = rand()%(N-1)+1) {
		// exchange x.list[0] and x.list[k]
		x.list[0] = x.list[0]^x.list[k];	// ^: XOR (1^0=1; 0^1=1; 1^1=0; 0^0=0)
		x.list[k] = x.list[k]^x.list[0];
		x.list[0] = x.list[0]^x.list[k];
	} // x.list: a random order of cities

	printf("%f\n",fitness(x) );


	/* Sample state space with 1% of runs to determine temperature schedule */
	for(i = 0,minf = 0,maxf = pow(10,10),x.fit=fitness(x);i < max(0.01*N,2);i++) {
		x = iterate(pow(10,10),x);
		minf = (x.fit < minf) ? x.fit : minf;
		maxf = (x.fit > maxf) ? x.fit : maxf;
	}

	printf("%f\t%f\n",minf,maxf );  // minf == maxf

	range = (maxf - minf)*PMELT;
	Dtemp = pow(TARGT,1.0/ITS);  // pow(0.01, 0.01): about 0.95

	//printf("%f\n", range*pow(Dtemp, 1)); 0.0

	/* Simulate Annealing */
	for(optgen = l = 1,optimum.fit = x.fit;l < ITS;l++) {
		x = iterate(range*pow(Dtemp,l),x);
		if(x.fit < optimum.fit) {
			optimum = x;
			optgen = l;
		}
		/* Reheat if stagnant */
		if(l-optgen == STAGFACTOR*ITS) 
			Dtemp = pow(Dtemp,.05*l/ITS);
		/* Graphics */
		printf("Iteration: %ld\t",l);
		printf("Fitness: %f\t\tTemp: %f\t\t",x.fit,range*pow(Dtemp,l));
		printf("Current Optimum %f found on %ld\t\t",optimum.fit,optgen);
		printf("Global Optimum is %d",N);
		printf("Sample Range: %f\tTemp decrement: %f\n\n",range,Dtemp);
		
		/* End Graphics */
	}
}
