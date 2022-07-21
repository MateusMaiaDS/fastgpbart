#include<iostream>
#include <vector>
#include <math.h>
#include<RcppEigen.h>
#include <algorithm>

using namespace std;



//[[Rcpp::export]]
// Inverting one matrix
Eigen::MatrixXd M_solve(const Eigen::MatrixXd M){
  int n(M.cols());
  Eigen::MatrixXd I =  Eigen::MatrixXd::Identity(n,n); // Creating a Identity matrix
  // return M.inverse(); // Solve the system Mx=I;
  return M.llt().solve(I);
}

// [[Rcpp::export]]
double get_log_D(Eigen::MatrixXd X){
  Eigen::VectorXd Dvec(X.ldlt().vectorD());
  return Dvec.array().log().sum();
}


// [[Rcpp::export]]
Eigen::MatrixXd MtM(const Eigen::MatrixXd& M){
  int n(M.cols()); // Getting the number of columns from x

  return Eigen::MatrixXd(n,n).setZero().selfadjointView<Eigen::Lower>().rankUpdate(M.adjoint());
}

// Getting the log of a MVN
//[[Rcpp::export]]
double fast_dmv(const Eigen::MatrixXd mu, const Eigen::MatrixXd M){

  int n = M.rows();
  Eigen::LLT<Eigen::MatrixXd> llt(M);
  return  -llt.matrixLLT().diagonal().array().log().sum() -0.5*MtM(llt.solve(mu))(0,0) -n*0.5*log(2*M_PI);
}

//
//   Eigen::MatrixXd I =  Eigen::MatrixXd::Identity(n,n); // Creating a Identity matrix
//   // return M.inverse(); // Solve the system Mx=I;
//   return M.llt().solve(I);
// }

mean_matrix <-as.matrix(rep(0,nrow(x_matrix)))


// Creating one sequence of values
vector<int> seq_along(int n){
  vector<int> vec_seq;
  for(int i =0; i<n;i++){
    vec_seq.push_back(i);
  }
  return vec_seq;
}


// Function to sample an integer from a sequence from
// 0:n
int sample_int(int n){
  return floor(R::runif(0,n));
}

// Sample an uniform value from a double vector
double sample_double(Eigen::VectorXd vec, int n_min_size){

  // Getting the range of the vector
  std::sort(vec.data(),vec.data()+vec.size() );
  return R::runif(vec[(n_min_size-1)],vec[(vec.size()-(n_min_size+1))]);

}



