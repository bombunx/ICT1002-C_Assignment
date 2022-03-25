#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double valueonly_coville4d( int dim, double *x){
    double x1 = x[0];
    double x2 = x[1];
    double x3 = x[2];
    double x4 = x[3];

    double term1 = 100 * (x1 * x1-x2) * (x1 * x1-x2);
    double term2 = (x1-1) * (x1-1);
    double term3 = (x3-1) * (x3-1);
    double term4 = 90 * (x3 * x3 - x4) * (x3 * x3 - x4);
    double term5 = 10.1 * ((x2-1) * (x2-1) + (x4-1)* (x4-1));
    double term6 = 19.8*(x2-1)*(x4-1);

    double y = term1 + term2 + term3 + term4 + term5 + term6;
}
double valueandderivatives_coville4d( int dim, double *x , double* grad, double *hessian_vecshaped){

    double y = valueonly_coville4d(dim,x);

    double x1 = x[0];
    double x2 = x[1];
    double x3 = x[2];
    double x4 = x[3];

    double term1 = 100 * (x1 * x1-x2) * (x1 * x1-x2);
    double term2 = (x1-1) * (x1-1);
    double term3 = (x3-1) * (x3-1);
    double term4 = 90 * (x3 * x3 - x4) * (x3 * x3 - x4);
    double term5 = 10.1 * ((x2-1) * (x2-1) + (x4-1)* (x4-1));
    double term6 = 19.8*(x2-1)*(x4-1);

    grad[0] = 100 * 2 * (x1 * x1-x2) * (x2) + 2 * (x1-1);
    grad[1] = 220.2 * x2 + 19.8 * x4 - 200 * x1 * x1 - 40;
    grad[2] = 2 * (x3-1) + 360 * (x3 * x3 - x4);
    grad[3] = 200.2 * x4 + 19.8 * x2 - 180 * x3 * x3 - 40;
}