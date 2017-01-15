/* mt19937ar.h */

#ifndef _MT19937AR_H_
#define _MT19937AR_H_

void init_genrand(unsigned long s);// initializes mt[N] with a seed
unsigned long genrand_int32();     // unsigned 32-bit integers.

long   genrand_int31();// unsigned 31-bit integers.
double genrand_real1();// uniform real in [0,1] (32-bit resolution). 
double genrand_real2();// uniform real in [0,1) (32-bit resolution). 
double genrand_real3();// uniform real in (0,1) (32-bit resolution).
double genrand_res53();// uniform real in [0,1) with 53-bit resolution.

#endif //_MT19937AR_H_
