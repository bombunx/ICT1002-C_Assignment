#include <stdio.h>
#include <stdlib.h>

double valueonly_beale2d( int dim, double *x
);
double valueandderivatives_beale2d( int dim, double *x , double* grad,
double *hessian_vecshaped
);

double valueonly_beale2d( int dim, double *x)
{
  if (dim !=2)
  {
    printf("dim is not 2, but %d\n",dim);
    exit(2);
  }
  
  double p1,p2,p3;
  
  p1= 1.5 - x[0] +x[0]*x[1];
  p2= 2.25 - x[0] +x[0]*x[1]*x[1]; 
  p3= 2.625 - x[0] +x[0]*x[1]*x[1]*x[1];   
  
  double ret = p1*p1 + p2*p2 + p3*p3;
  
  return ret;
}

double valueandderivatives_beale2d( int dim, double *x , double* grad,
double *hessian_vecshaped)
{
  if (dim !=2)
  {
    printf("dim is not 2, but %d\n",dim);
    exit(2);
  }
    
  if (grad == NULL)
  {
    printf("valueandderivatives_beale2d: grad == NULL\n");
    exit(10);  
  }
  if (hessian_vecshaped == NULL)
  {
    printf("valueandderivatives_beale2d: hessian_vecshaped == NULL\n");
    exit(11);  
  }
  
  double ret  = valueonly_beale2d( dim, x);
  
  double p1,p2,p3;
  
  
  //gradient
  p1= 1.5 - x[0] +x[0]*x[1];
  p2= 2.25 - x[0] +x[0]*x[1]*x[1]; 
  p3= 2.625 - x[0] +x[0]*x[1]*x[1]*x[1]; 
  
  grad[0] = 2*p1*(-1+x[1]) + 2*p2*(-1+x[1]*x[1])  + 2*p3*(-1+x[1]*x[1]*x[1]); 
  grad[1] = 2*p1*x[0] +  2*p2*2*x[0]*x[1] + 2*p3*3*x[0]*x[1]*x[1]; 

  //Hessian  
  double q1,q2,q3;
  q1 = -1+x[1];
  q2 = -1+x[1]*x[1];
  q3 = -1+x[1]*x[1] *x[1];  
  
  hessian_vecshaped[0+2*0] = 2*q1*q1 + 2*q2*q2 + 2*q3*q3;  
  hessian_vecshaped[1+2*1] = 2*x[0]*x[0] 
                          + 8*x[0]*x[0]*x[1]*x[1] + 2*p2*2*x[0] 
                          + 18*x[0]*x[0]*x[1]*x[1]*x[1]*x[1] + 2*p3*6*x[0]*x[1];
  
  hessian_vecshaped[1+2*0] = 2*x[0]*q1 +2*p1 + 4*x[0]*x[1]*q2 + 2*p2*2*x[1]
                          + 6*x[0]*x[1]*x[1]*q3 + 2*p3*3*x[1]*x[1];
  hessian_vecshaped[0+2*1] = hessian_vecshaped[1+2*0];                        
  return ret;
  
}



int main()
{
  //plotovergrid2d( "./griddata.txt");
  return 0;
}
