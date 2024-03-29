#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_linalg.h>


// Function Functions
double valueonly_beale2d( int dim, double *x);
double valueandderivatives_beale2d( int dim, double *x , double* grad, double *hessian_vecshaped);

double valueonly_matya2d( int dim, double *x);
double valueandderivatives_matya2d( int dim, double *x , double* grad, double *hessian_vecshaped);

// Function surface 
void func_surface_plot(int dim, double* grad, double *hessian_vecshaped, const double max_range, const double min_range, double (*)(int,double*,double*,double*));

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
  return y;
}
double valueandderivatives_coville4d( int dim, double *x , double* grad, double *hessian_vecshaped){

    double y = valueonly_coville4d(dim,x);

    double x1 = x[0];
    double x2 = x[1];
    double x3 = x[2];
    double x4 = x[3];

    grad[0] = 100 * 2 * (x1 * x1-x2) * (x2) + 2 * (x1-1);
    grad[1] = 220.2 * x2 + 19.8 * x4 - 200 * x1 * x1 - 40;
    grad[2] = 2 * (x3-1) + 360 * (x3 * x3 - x4);
    grad[3] = 200.2 * x4 + 19.8 * x2 - 180 * x3 * x3 - 40;

    return y;
}



// Gradient Descent Functions
double gradient_descent_simple(int dim, double function, double *grad, double *x, double *hess, double alpha, double threshold, int max_iter, double (*)(int,double*,double*,double*));
double gradient_descent_armijo(int dim, double function, double *grad, double *x, double *hess, double alpha,  double momentum, double threshold, int max_iter, double (*)(int,double*,double*,double*));
double gradient_descent_newton(int dim, double function, double *grad, double*x, double *hess, double threshold, double epsilon, int max_iter, double (*)(int,double*,double*,double*));

int main()
{
  // User input
  int input;
  int dim, seed;
  double max_range;
  double min_range;

  printf("Enter dimensions: ");
  scanf("%d", &dim);

  printf("Enter random number generator seed: ");
  scanf("%d", &seed);

  printf("Enter max range: ");
  scanf("%lf", &max_range);

  printf("Enter min range: ");
  scanf("%lf", &min_range);

  double * x = malloc (dim * sizeof (double));
  double * grad = calloc (dim, sizeof (double));
  double * hessian = calloc (dim * dim, sizeof(double));
  double function;

  double(*test_function)(int,double*,double*,double*);
  test_function = &valueandderivatives_matya2d;

  printf("Select algorithm: \n [0] Simple Gradient Descent \n [1] Momentum Gradient Descent \n [2] Gradient Descent with Newton's Method \n");

  scanf("%d", &input);
  printf("You entered: %d\n", input);

  if (input == 0){
    // Simple Gradient Descent
    printf("Simple Gradient Descent \n");

    double alpha;
    double threshold;
    int iteration;

    printf("Enter alpha: ");
    scanf("%lf", &alpha);

    printf("Enter threshold: ");
    scanf("%lf", &threshold);

    printf("Enter iteration: ");
    scanf("%d", &iteration);

    printf("dim is %d, alpha is %lf, max range is %lf, min range is %lf \n", dim, alpha, max_range, min_range);

    // initialise starting point
    srand (seed); // seed random number generator
    for(int i =0;i < dim; i++){
      x[i] = fmod(rand(),(max_range-min_range+1)) + min_range;
    }
    function = (*test_function)(dim,x,grad,hessian);

    // Parameters for Matya2D Function -- Gradient Descent Simple
    // dim = 4, range = [0.0,1.0], seed = 5, alpha = 0.5, threshold = 1e-5, max_iter = 1000

    //function = valueandderivatives_coville4d(dim,x,grad,hessian);

    // Parameters for Colville4D Function -- Gradient Descent Simple
    // dim = 4, range = [1.001,1.005], seed = 10, alpha = 0.001075, threshold = 1e-5, max_iter = 10000
    
    // gradient_descent_simple(dim,function,grad,x,hessian,0.0175,1e-5,1000);
    gradient_descent_simple(dim,function,grad,x,hessian,alpha,threshold,iteration,test_function);
  }

  else if (input == 1){
    // Momentum Gradient Descent
    printf("Momentum Gradient Descent \n");

    double alpha;
    double momentum;
    double threshold;
    int iteration;

    printf("Enter alpha: ");
    scanf("%lf", &alpha);

    printf("Enter momentum: ");
    scanf("%lf", &momentum);

    printf("Enter threshold: ");
    scanf("%lf", &threshold);

    printf("Enter iteration: ");
    scanf("%d", &iteration);

    srand (seed); // seed random number generator
    for(int i =0;i < dim; i++){
      x[i] = fmod(rand(),(max_range-min_range+1)) + min_range;
    }
    function = (*test_function)(dim,x,grad,hessian);
    gradient_descent_armijo(dim,function,grad,x,hessian,alpha,momentum,threshold,iteration, test_function); 
  }

  else if (input == 2){
    // Gradient Descent with Newton's Method
    printf("Gradient Descent with Newton's Method \n");

    double threshold;
    double epsilon;
    int iteration;

    printf("Enter epsilon: ");
    scanf("%lf", &epsilon);

    printf("Enter threshold: ");
    scanf("%lf", &threshold);

    printf("Enter iteration: ");
    scanf("%d", &iteration);

    printf("dim is %d, threshold is %lf, max range is %lf, min range is %lf \n", dim, threshold, max_range, min_range);

    
    // initialise starting point
    // srand (5); // seed random number generator
    // for(int i =0;i < dim; i++){
    //   x[i] = max_range * ((1.0 * rand()) / RAND_MAX) + min_range;
    // }
    // x[0] = 2;
    // x[1] = 0;
    // function = valueandderivatives_beale2d(dim,x,grad,hessian);

    srand (seed); // seed random number generator
    for(int i =0;i < dim; i++){
      x[i] = fmod(rand(),(max_range-min_range+1)) + min_range;
    }
    function = (*test_function)(dim,x,grad,hessian);
    // gradient_descent_newton(dim,function,grad,x,hessian,1e-10,1e-7,1000);
    gradient_descent_newton(dim,function,grad,x,hessian,threshold,epsilon,iteration,test_function);
  }
  
  func_surface_plot(dim, grad, hessian, max_range, min_range, test_function);
  free(x);
  free(grad);
  free(hessian);
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


void func_surface_plot(int dim, double* grad, double *hessian_vecshaped, const double max_range, const double min_range, double (*functionPointer)(int dim, double *x, double* grad,double* hess)){
  double * x = malloc (dim * sizeof (double));
  double function;

  FILE *out_file; // output file
  out_file = fopen ("funcSurface.txt", "w");

  // x_1 value
  fprintf(out_file,"x_1 = [");

  for (double i = min_range; i<max_range; i+=0.5) {
    fprintf(out_file,"%.6f ", i);
  }
  fprintf(out_file,"]\n");

  // x_2 value
  fprintf(out_file,"x_2 = [");

  for (double i = min_range; i<max_range; i+=0.5) {
    fprintf(out_file,"%.6f ", i);
  }
  fprintf(out_file,"]\n");

  // func / y value
  for (double i = min_range; i<max_range; i+=0.5) {
    x[1] = i;
    fprintf(out_file,"y = [");
    for (double j = min_range; j<max_range; j+=0.5){
      x[0] = j;
      function = (*functionPointer)(dim,x,grad,hessian_vecshaped);
      fprintf(out_file,"%.6f ", function);
    }
    fprintf(out_file,"]\n");
  }
  
  fclose(out_file); // Close the output file after loop ends
}



double gradient_descent_simple(int dim, double function, double *grad, double *x, double *hess, double alpha, double threshold, int max_iter, double (*functionPointer)(int dim, double *x, double* grad,double* hess)){

  double sum_grad_squared;
  double effective_vector;
  //printf("x-values: %f \t %f \t %f \t %f \n y-value: %f \n gradient: %f \n",x[0],x[1],x[2],x[3],function,effective_vector);
  sum_grad_squared = 0;
  for (int counter = 0; counter < dim; counter++){
    sum_grad_squared += (grad[counter] * grad[counter]);
  }
  effective_vector = sqrt(sum_grad_squared);  


  int num_iter =0;
  bool success = true;

  FILE *out_file; // output file
  out_file = fopen ("output.txt", "w");

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
  printf("vector = %.6f\n",effective_vector);  
  
  while (num_iter < max_iter){
    
    // new x values
    for (int counter = 0; counter < dim; counter++){
      x[counter] = x[counter] + (alpha * grad[counter] * -1);
    }
    // new y values
    function = (*functionPointer)(dim,x,grad,hess);
    // function=valueandderivatives_coville4d(4,x,grad,hess);
    
    num_iter++;

    sum_grad_squared = 0;
    for (int counter = 0; counter < dim; counter++){
    sum_grad_squared += (grad[counter] * grad[counter]);
    }
    effective_vector = sqrt(sum_grad_squared);

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
    printf("vector = %.6f\n",effective_vector);

    if (effective_vector < threshold) {
      break;
    }

    if ((isnan(effective_vector) || (isinf(effective_vector)))){
      success = false;
      break;
    }
  }

  fclose(out_file); // Close the output file after loop ends

  // print results
  if (success == false || num_iter == max_iter){
    printf("Gradient descent does not converge.\n");
    printf("Effective Vector: %.6f",effective_vector);

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
  return effective_vector;
}

double gradient_descent_armijo(int dim, double function, double *grad, double *x, double *hess, double alpha, double momentum, double threshold, int max_iter, double (*functionPointer)(int dim, double *x, double* grad,double* hess)){
{
  double sum_grad_squared;
  double effective_vector;
  double *change = calloc(dim, sizeof(double));

  double *new_change = calloc(dim, sizeof(double));


  //printf("x-values: %f \t %f \t %f \t %f \n y-value: %f \n gradient: %f \n",x[0],x[1],x[2],x[3],function,effective_vector);
  sum_grad_squared = 0;
  for (int counter = 0; counter < dim; counter++){
    sum_grad_squared += (grad[counter] * grad[counter]);
  }
  effective_vector = sqrt(sum_grad_squared);  


  int num_iter =0;
  bool success = true;

  FILE *out_file; // output file
  out_file = fopen ("output.txt", "w");

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
  printf("vector = %.6f\n",effective_vector);  
  
  while (num_iter < max_iter){
    
    for (int i=0; i <dim;i++){
      new_change[i] = alpha * grad[i] + momentum * change[i];
    }
    

    for (int counter = 0; counter < dim; counter++){
      x[counter] = x[counter] - new_change[counter];
    }

    change = new_change;
    function = functionPointer(dim,x,grad,hess);

    num_iter++;

    sum_grad_squared = 0;
    for (int counter = 0; counter < dim; counter++){
    sum_grad_squared += (grad[counter] * grad[counter]);
    }
    effective_vector = sqrt(sum_grad_squared);

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
    printf("vector = %.6f\n",effective_vector);

    if (effective_vector < threshold) {
      break;
    }

    if (isnan(effective_vector) || (isinf(effective_vector))){
      success = false;
      break;
    }    
  }
  fclose(out_file); // Close the output file after loop ends

  // print results
  if (success == false || num_iter == max_iter){
    printf("Gradient descent does not converge.\n");
    printf("%.6f",effective_vector);

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
  return effective_vector;
}
}

double gradient_descent_newton(int dim, double function, double *grad, double*x, double *hess, double threshold, double epsilon, int max_iter, double (*functionPointer)(int dim, double *x, double* grad,double* hess)){

  
  double sum_grad_squared;
  for (int counter = 0; counter < dim; counter++){
    sum_grad_squared += (grad[counter] * grad[counter]);
  }
  double effective_vector = sqrt(sum_grad_squared); 

  int num_iter =0;
  double *negative_grad = calloc (dim, sizeof (double));
  bool success = true;

  double *A = calloc (dim * dim, sizeof(double));

  gsl_matrix *gsl_hessian = gsl_matrix_calloc(dim,dim);
  gsl_vector *gsl_grad = gsl_vector_calloc(dim);
  gsl_vector *gsl_x = gsl_vector_calloc(dim);
  gsl_matrix *mA = gsl_matrix_calloc(dim,dim);
  gsl_vector *temp = gsl_vector_calloc(dim);

  FILE *out_file; // output file
  out_file = fopen ("output.txt", "w");

  while (num_iter < max_iter){

    gsl_matrix_view gsl_hessian_view = gsl_matrix_view_array(hess,dim,dim);
    gsl_hessian = &(gsl_hessian_view.matrix);

    gsl_vector_view gsl_grad_view = gsl_vector_view_array(negative_grad,dim);
    gsl_grad = &(gsl_grad_view.vector);

    gsl_vector_view gsl_x_view = gsl_vector_view_array(x,dim);
    gsl_x = &(gsl_x_view.vector);

    gsl_matrix_view mA_view = gsl_matrix_view_array(A,dim,dim);
    mA = &(mA_view.matrix);

    gsl_matrix_set_identity(mA);
    gsl_matrix_scale(mA,epsilon); 
    gsl_matrix_add(mA,gsl_hessian);


    gsl_blas_dgemv(CblasNoTrans,1.0,mA,gsl_x,0.0,temp);
    gsl_vector_add(temp,gsl_grad);
    gsl_linalg_HH_solve(mA,temp,gsl_x);

    for (int i = 0; i < dim; i++) {
      x[i] = gsl_vector_get(gsl_x,i); // new x values
    }
    function = functionPointer(dim,x,grad,hess);    
    num_iter++;


    for (int counter = 0; counter < dim; counter++){
      negative_grad[counter] = grad[counter] * -1;
    }

    sum_grad_squared = 0;
    for (int counter = 0; counter < dim; counter++){
    sum_grad_squared += (grad[counter] * grad[counter]);
    }
    effective_vector = sqrt(sum_grad_squared);

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
    printf("vector = %.6f\n",effective_vector);

    if (effective_vector < threshold) {
      break;
    }

    if ((isnan(effective_vector) || (isinf(effective_vector)))){
      success = false;
      break;
    }
  }
  fclose(out_file); // Close the output file after loop ends

  // print results
  if (success == false || num_iter == max_iter){
    printf("Gradient descent does not converge.\n");
    printf("%.6f",effective_vector);

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
  return effective_vector;
}