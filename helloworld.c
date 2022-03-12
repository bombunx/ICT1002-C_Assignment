#include <stdio.h>


double valueonly(int dim, double *x);
    // computes f(x) and returns minimum

double valueandderivatives(int dim, double *x, double *grad, double *hessian_vecshaped);
    // computes f(x) as return value and gradient delta(f(x)) and Hessian[f](x) in pointers

double *x;
    // input value (x) to compute f(x)
    // x[k] is the component xk of the input vector for x.

int dim;
    // number of components in that vector

double* grad;
    // pointer to an array with dim elements
    // grad[k] contains computed gradient of derivate x over xk
    
double *hessian_vecshaped;
    // should be a pointer to an array with dim*dim elements.


int main(){

    printf("Hello world!\n");
    printf("Welcome to ICT1002!\n");

    return 0;
}