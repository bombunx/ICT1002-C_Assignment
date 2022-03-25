#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>


// Function Functions
double valueonly_beale2d( int dim, double *x);
double valueandderivatives_beale2d( int dim, double *x , double* grad, double *hessian_vecshaped);

double valueonly_matya2d( int dim, double *x);
double valueandderivatives_matya2d( int dim, double *x , double* grad, double *hessian_vecshaped);


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

// Gradient Descent Functions
double gradient_descent_simple(int dim, double function, double *grad, double *x, double *hess, double alpha, double threshold, int max_iter);

int main()
{
  // User input
  int input;
  int dim = 2;
  double max_range = 10;
  double min_range = -10;

  double * x = malloc (dim * sizeof (double));
  double * grad = calloc (dim, sizeof (double));
  double * hessian = calloc (dim * dim, sizeof(double));
  double function;

  
  printf("Select algorithm: \n [0] Simple Gradient Descent \n [1] Momentum Gradient Descent \n [2] Gradient Descent with Newton's Method \n");

  scanf("%d", &input);
  printf("You entered: %d\n", input);

  if (input == 0){
    // Simple Gradient Descent
    printf("Simple Gradient Descent \n");
    printf("Set parameters (dim, alpha): \n"); // dim, alpha
    printf("Max range, Min range: \n"); // max_range, min_range

    
    // initialise starting point
    srand (5); // seed random number generator
    for(int i =0;i < dim; i++){
      x[i] = max_range * ((1.0 * rand()) / RAND_MAX) + min_range;
    }

    function = valueandderivatives_matya2d(dim,x,grad,hessian);
    // Parameters for Matya2D Function -- Gradient Descent Simple
    // dim = 4, range = [0.0,1.0], seed = 5, alpha = 0.5, threshold = 1e-5, max_iter = 1000

    //function = valueandderivatives_coville4d(dim,x,grad,hessian);

    // Parameters for Colville4D Function -- Gradient Descent Simple
    // dim = 4, range = [1.001,1.005], seed = 10, alpha = 0.001075, threshold = 1e-5, max_iter = 10000
    
    gradient_descent_simple(dim,function,grad,x,hessian,0.5,1e-5,1000);
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
  
  double ret  = valueonly_beale2d(dim, x);
  
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
  //printf("x-values: %f \t %f \t %f \t %f \n y-value: %f \n gradient: %f \n",x[0],x[1],x[2],x[3],function,norm_grad);

  int num_iter =0;
  double *negative_grad = calloc (dim, sizeof (double));
  bool success = true;

  FILE *out_file; // output file
  out_file = fopen ("output.txt", "w");

  while (num_iter < max_iter){

    for (int counter = 0; counter < dim; counter++){
    negative_grad[counter] = grad[counter] * -1;
    }

    // new y values
    function = valueandderivatives_matya2d(2,x,grad,hess);
    // function=valueandderivatives_coville4d(4,x,grad,hess);

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

    //Loop to output results at each iteration and save to file at the same time

    printf("Iteration: %d \t y = %.6f \t",num_iter,function);
    fprintf(out_file,"Iteration: %d \t y = %.6f \t",num_iter,function);

    printf("x =[");
    fprintf(out_file,"x =[");

    for (int i = 0; i < dim;i++){
      if (i != (dim-1)){
        printf(" %.6f, ",x[i]);
        fprintf(out_file," %.6f, ",x[i]);
      }
      else{
        printf(" %.6f ]\t",x[i]);
        fprintf(out_file," %.6f ]\n",x[i]);
      }
    }
    printf("vector = %.6f\n",norm_grad);


    if (norm_grad < threshold) {
      break;
    }

    if ((isnan(norm_grad) || (isinf(norm_grad)))){
      success = false;
      break;
    }
  }

  fclose(out_file); // Close the output file after loop ends
  // print results
  if (success == false || num_iter == max_iter){
    printf("Gradient descent does not converge.\n");
    printf("%.6f",norm_grad);

    FILE *out_file; // If it does not converge, rewrite the file to show no output
    out_file = fopen ("output.txt", "w");
    fprintf(out_file,"Gradient descent does not converge.\n");
    fclose(out_file);
  }

  else{
    // Solution
    printf("Solution: y = %f \t ",function);
    printf("x =[");
    for (int i = 0; i < dim;i++){
      if (i != (dim-1)){
        printf(" %.6f, ",x[i]);
      }
      else{
        printf(" %.6f ]\t",x[i]);
      }
    }

  }
}
