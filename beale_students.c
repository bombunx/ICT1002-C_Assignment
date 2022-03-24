#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Function Functions
double valueonly_beale2d( int dim, double *x);
double valueandderivatives_beale2d( int dim, double *x , double* grad, double *hessian_vecshaped);

double valueonly_matya2d( int dim, double *x);
double valueandderivatives_matya2d( int dim, double *x , double* grad, double *hessian_vecshaped);

// Gradient Descent Functions
double gradient_descent_simple(int dim, double function, double *grad, double *x, double *hess, double alpha, double threshold, int max_iter);

int main()
{
  int dim = 2;
  double * x = malloc (dim * sizeof (double));
  double * grad = calloc (dim, sizeof (double));
  double * hessian = calloc (dim * dim, sizeof(double));
  int input;
  double function;


  // initialise starting point
  x[0] = 1;
  x[1] = 1;

  function = valueandderivatives_matya2d(dim,x,grad,hessian);
  
  
  printf("Select algorithm: \n [0] Simple Gradient Descent \n [1] Momentum Gradient Descent \n [2] Gradient Descent with Newton's Method \n");

  scanf("%d", &input);
  printf("You entered: %d\n", input);

  if (input == 0){
    // Simple Gradient Descent
    printf("Simple Gradient Descent \n");
    //printf("Set parameters \n");
    
    gradient_descent_simple(dim,function,grad,x,hessian,7,1e-5,1000);
    free(x);
    free(grad);
    free(hessian);
  }

  else if (input == 1){
    // Momentum Gradient Descent
    printf("Momentum Gradient Descent \n");
  }

  else if (input == 2){
    // Gradient Descent with Newton's Method
    printf("Gradient Descent with Newton's Method \n");
  }
  
  //plotovergrid2d( "./griddata.txt");
  return 0;
}

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
double valueandderivatives_beale2d( int dim, double *x , double* grad, double *hessian_vecshaped)
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



double valueonly_matya2d( int dim, double *x){
    if (dim !=2)
  {
    printf("dim is not 2, but %d\n",dim);
    exit(2);
  }
  
  double p1,p2;
  
  p1= 0.26 * (x[0]*x[0] +x[1]*x[1]);
  p2= 0.48 * x[0] *x[1]; 
  
  double ret = p1 - p2;
  
  return ret;
}
double valueandderivatives_matya2d( int dim, double *x , double* grad, double *hessian_vecshaped){
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
  
  double ret  = valueonly_matya2d(dim, x);
  
  double p1;
  
  
  //gradient

  
  grad[0] = 0.52 * x[0] - 0.48 * x[1];
  grad[1] = 0.52 * x[1] - 0.48 * x[0];

  //Hessian  
  hessian_vecshaped[0+2*0] = 0.52;
  hessian_vecshaped[1+2*1] = 0.52;
  hessian_vecshaped[1+2*0] = -0.48;
  hessian_vecshaped[0+2*1] = hessian_vecshaped[1+2*0];                        
  return ret;
}


double gradient_descent_simple(int dim, double function, double *grad, double *x, double *hess, double alpha, double threshold, int max_iter){
  

  double sum_grad_squared;
  for (int counter = 0; counter < dim; counter++){
    sum_grad_squared += (grad[counter] * grad[counter]);
  }
  double norm_grad = sqrt(sum_grad_squared);
  //printf("x-values: %f \t %f \n y-value: %f \n gradient: %f \n",x[0],x[1],function,norm_grad);

  int num_iter =0;
  double *negative_grad = calloc (dim, sizeof (double));
  
  while (num_iter < max_iter){

    for (int counter = 0; counter < dim; counter++){
    negative_grad[counter] = grad[counter] * -1;
    }

    // new y values
    function = valueandderivatives_matya2d(2,x,grad,hess);
  
    // new x values
    for (int counter = 0; counter < dim; counter++){
      x[counter] = x[counter] + (alpha * negative_grad[counter]);
    }

    sum_grad_squared = 0;
    for (int counter = 0; counter < dim; counter++){
      sum_grad_squared += (grad[counter] * grad[counter]);
    }
    norm_grad = sqrt(sum_grad_squared);

    num_iter++;

    // Need a loop to output all the X values
    printf("Iteration: %d \t y = %.6f \t x = %.6f \t %.6f \t gradient: %.6f \n",num_iter,function,x[0],x[1],norm_grad);
    if (norm_grad < threshold){
      break;
    }
  }

  // print results
  if (num_iter == max_iter){
    printf("Gradient descent does not converge.\n");
  }

  else{
    printf("Solution: y = %f \t x = [%f,%f]\n",function,x[0],x[1]);
  }
}
