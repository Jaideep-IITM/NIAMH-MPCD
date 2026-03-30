#ifndef RAND_H
#define RAND_H

/* ****************************************** */
/* ****************************************** */
/* ****************************************** */
/* ************ DECLARE FUNCTIONS *********** */
/* ****************************************** */
/* ****************************************** */
/* ****************************************** */

// Methods exclusive to specific RNGs are not prototyped here

unsigned long RandomSeedSRD (unsigned long seed);
void resetBMSpare(void);

void init_genrand( unsigned long s );
void init_by_array( unsigned long init_key[], int key_length );
unsigned long genrand_int32( void );
long genrand_int31( void );
double genrand_real( void );
double genrand_pmOne( void );
void genrand_coneNP( double vec[],double theta,int dimension );
void genrand_cone( double axis[],double vecOut[],double theta,int dimension );

float genrand_gauss( void );
double genrand_gaussMB(double KBT,double M);
double genrand_gaussGen(double mu,double sigma);
double genrand_exp(double lambda);
int genrand_poisson(double lambda);
float genrand_rayleigh( float std );
void genrand_sphere( double vec[],int dimension );

double *ranvec3D( double *v );
double *ranvec2D( double *v );
double *ranvec( double *v,int dimension );
double *ranshift( double *v,int doShift,int dimension );

int rand_particle( int POP );

#endif
