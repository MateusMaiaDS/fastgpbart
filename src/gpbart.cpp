#define _USE_MATH_DEFINES
#include<Rcpp.h>
#include<RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#include <vector>
#include<cmath>
#include <math.h>
#include "tree.h"

// The line above (depends) it will make all the dependcies be included on the file
using namespace Rcpp;
using namespace RcppEigen;
using namespace std;

using Eigen::Map;
using Eigen::LLT;
using Eigen::Lower;
using Eigen::Upper;


//[[Rcpp::export]]
Eigen::MatrixXd matrix_slice(Eigen::MatrixXd &x,
                             vector<int> &indexes){

  // Creating a matrix with the same dimensions of the x
  Eigen::MatrixXd slice_matrix(indexes.size(),x.cols());

  // Setting the new matrix elements
  for(int i=0;i<indexes.size();i++){
    slice_matrix.row(i) = x.row(indexes[i]);
  }

  return slice_matrix;

}

//[[Rcpp::export]]
int initialize_test(Eigen::MatrixXd x_train,Eigen::MatrixXd x_test,
                    Eigen::VectorXd residuals,
                    double phi, double nu,
                    double tau, double tau_mu, int node_min_size){

  // Defining initial values for omega matrices
  Eigen::MatrixXd i_omega,i_omega_plus_tau,i_omega_plus_tau_inv;

  i_omega = omega(x_train,phi,nu);
  i_omega_plus_tau = omega_plus_tau(x_train,phi,nu,tau);
  i_omega_plus_tau_inv = M_solve(i_omega_plus_tau);

  // Getting
  int n_obs_train, n_obs_test;
  n_obs_train = x_train.rows();
  n_obs_test = x_test.rows();

  // gpb_Tree tree_example(n_obs_train,n_obs_test,i_omega,i_omega_plus_tau,i_omega_plus_tau_inv);

  // tree_example.DisplayNodes();

  // tree_example.list_node[0].prev_omega_plus_tau_inv();
  return 0;

}


// // [[Rcpp::depends(RcppEigen)]]
// Tree gpb_grow(gpb_Tree curr_tree,
//           Eigen::MatrixXd x_train,
//           Eigen::MatrixXd x_test,
//           // Adding the parameters necessary for the GP-BART scheme
//           double phi,
//           double nu,
//           double tau,
//           int n_min_size,
//           int *id_t, // Object to identify if the same tree was returned
//           int *verb_node_index // Store the node which was growned
// ){
//
//   // Declaring variables
//   bool bad_tree=true;
//   int bad_tree_counter = 0;
//
//   // Getting information from x
//   int p = x_train.cols();
//   int n_terminals;
//   int n_nodes = curr_tree.list_node.size();
//   int g_node, g_var, n_train, n_test;
//   double g_rule;
//
//   // Getting observations that are on the left and the ones that are in the right
//   vector<int> new_left_train_index;
//   vector<int> new_right_train_index;
//   vector<int> curr_obs_train; // Observations that belong to that terminal node
//
//   // Same from above but for test observations
//   // Getting observations that are on the left and the ones that are in the right
//   vector<int> new_left_test_index;
//   vector<int> new_right_test_index;
//   vector<int> curr_obs_test; // Observations that belong to that terminal node
//
//   // Getting terminal nodes
//   vector<gpb_node> t_nodes = curr_tree.getTerminals();
//   n_terminals = t_nodes.size();
//
//   while(bad_tree){
//
//     // Sampling one of the nodes (this function will sample the node index from
//     // the vector)
//     g_node = sample_int(n_terminals);
//     g_var = sample_int(p);
//
//     // Number of observations in the node that will grow
//     curr_obs_train = t_nodes[g_node].obs_train; // Current observations on that node
//     n_train = t_nodes[g_node].obs_train.size();
//
//     // Doing the same but for the test
//     curr_obs_test = t_nodes[g_node].obs_test; // Current observations on that node
//     n_test = t_nodes[g_node].obs_test.size();
//
//     // Do not let split if is not possible to split
//     if( floor(n_train/2) <= n_min_size){
//       *id_t = 1;
//       break;
//     }
//
//     g_rule = sample_double(x_train.col(g_var),n_min_size);
//
//     // Iterating over the train observations
//     for(int i = 0; i<n_train; i++){
//       if(x_train.coeff(curr_obs_train[i],g_var)<=g_rule){
//         new_left_train_index.push_back(curr_obs_train[i]);
//       } else {
//         new_right_train_index.push_back(curr_obs_train[i]);
//       }
//     }
//
//     // Iterating over the test observations
//     for(int i = 0; i<n_test; i++){
//       if(x_test.coeff(curr_obs_test[i],g_var)<=g_rule){
//         new_left_test_index.push_back(curr_obs_test[i]);
//       } else {
//         new_right_test_index.push_back(curr_obs_test[i]);
//       }
//     }
//
//     if(new_right_train_index.size()>=n_min_size && new_left_train_index.size()>=n_min_size){
//
//       // Getting the new nodes
//       gpb_node new_node_left(n_nodes,new_left_train_index,
//                          new_left_test_index,
//                          -1,-1,t_nodes[g_node].depth+1,
//                          g_var,g_rule,t_nodes[g_node].mu,
//                          omega(x_train.row(new_left_test_index),phi,nu));
//
//       gpb_node new_node_right(n_nodes+1,new_right_train_index,
//                           new_right_test_index,
//                           -1,-1,t_nodes[g_node].depth+1,
//                           g_var,g_rule,t_nodes[g_node].mu);
//
//       // Adding the new nodes
//       curr_tree.list_node.push_back(new_node_left);
//       curr_tree.list_node.push_back(new_node_right);
//
//       // Transforming the growned node into non-terminal
//       curr_tree.list_node[t_nodes[g_node].index].left = n_nodes;
//       curr_tree.list_node[t_nodes[g_node].index].right = n_nodes+1;
//
//       // Storing the index of the node index
//       *verb_node_index = t_nodes[g_node].index;
//
//       // Success to create a good tree;
//       bad_tree = false;
//
//     } else {
//
//       // Cleaning the vectors
//       new_left_train_index.clear();
//       new_right_train_index.clear();
//       curr_obs_train.clear();
//       new_left_test_index.clear();
//       new_right_test_index.clear();
//       curr_obs_test.clear();
//
//       bad_tree_counter++;
//
//       // Avoid infinite loop
//       if(bad_tree_counter==2){
//         bad_tree = false;
//         *id_t = 1;
//         // cout << " BAD TREE" << endl;
//       }
//     }
//
//
//   }
//
//   return curr_tree;
//
//
// }


//========== BART FUNCTIONS ==============



// [[Rcpp::depends(RcppEigen)]]
Tree grow_bart(Tree curr_tree,
          Eigen::MatrixXd x_train,
          Eigen::MatrixXd x_test,
          int n_min_size,
          int *id_t, // Object to identify if the same tree was returned
          int *verb_node_index // Store the node which was growned
){

  // Declaring variables
  bool bad_tree=true;
  int bad_tree_counter = 0;

  // Getting information from x
  int p = x_train.cols();
  int n_terminals;
  int n_nodes = curr_tree.list_node.size();
  int g_node, g_var, n_train, n_test;
  double g_rule;

  // Getting observations that are on the left and the ones that are in the right
  vector<int> new_left_train_index;
  vector<int> new_right_train_index;
  vector<int> curr_obs_train; // Observations that belong to that terminal node

  // Same from above but for test observations
  // Getting observations that are on the left and the ones that are in the right
  vector<int> new_left_test_index;
  vector<int> new_right_test_index;
  vector<int> curr_obs_test; // Observations that belong to that terminal node

  // Getting terminal nodes
  vector<node> t_nodes = curr_tree.getTerminals();
  n_terminals = t_nodes.size();

  while(bad_tree){

    // Sampling one of the nodes (this function will sample the node index from
    // the vector)
    g_node = sample_int(n_terminals);
    g_var = sample_int(p);

    // Number of observations in the node that will grow
    curr_obs_train = t_nodes[g_node].obs_train; // Current observations on that node
    n_train = t_nodes[g_node].obs_train.size();

    // Doing the same but for the test
    curr_obs_test = t_nodes[g_node].obs_test; // Current observations on that node
    n_test = t_nodes[g_node].obs_test.size();

    // Do not let split if is not possible to split
    if( floor(n_train/2) <= n_min_size){
      *id_t = 1;
      break;
    }

    g_rule = sample_double(x_train.col(g_var),n_min_size);

    // Iterating over the train observations
    for(int i = 0; i<n_train; i++){
      if(x_train.coeff(curr_obs_train[i],g_var)<=g_rule){
        new_left_train_index.push_back(curr_obs_train[i]);
      } else {
        new_right_train_index.push_back(curr_obs_train[i]);
      }
    }

    // Iterating over the test observations
    for(int i = 0; i<n_test; i++){
      if(x_test.coeff(curr_obs_test[i],g_var)<=g_rule){
        new_left_test_index.push_back(curr_obs_test[i]);
      } else {
        new_right_test_index.push_back(curr_obs_test[i]);
      }
    }

    if(new_right_train_index.size()>=n_min_size && new_left_train_index.size()>=n_min_size){

      // Getting the new nodes
      node new_node_left(n_nodes,new_left_train_index,
                         new_left_test_index,
                         -1,-1,t_nodes[g_node].depth+1,
                         g_var,g_rule,t_nodes[g_node].mu);

      node new_node_right(n_nodes+1,new_right_train_index,
                          new_right_test_index,
                          -1,-1,t_nodes[g_node].depth+1,
                          g_var,g_rule,t_nodes[g_node].mu);

      // Adding the new nodes
      curr_tree.list_node.push_back(new_node_left);
      curr_tree.list_node.push_back(new_node_right);

      // Transforming the growned node into non-terminal
      curr_tree.list_node[t_nodes[g_node].index].left = n_nodes;
      curr_tree.list_node[t_nodes[g_node].index].right = n_nodes+1;

      // Storing the index of the node index
      *verb_node_index = t_nodes[g_node].index;

      // Success to create a good tree;
      bad_tree = false;

    } else {

      // Cleaning the vectors
      new_left_train_index.clear();
      new_right_train_index.clear();
      curr_obs_train.clear();
      new_left_test_index.clear();
      new_right_test_index.clear();
      curr_obs_test.clear();

      bad_tree_counter++;

      // Avoid infinite loop
      if(bad_tree_counter==2){
        bad_tree = false;
        *id_t = 1;
        // cout << " BAD TREE" << endl;
      }
    }


  }

  return curr_tree;


}

// Prune a tree
// [[Rcpp::depends(RcppEigen)]]
Tree prune_bart(Tree curr_tree, int* id_t, int* verb_node_index){

  // New list of nodes
  int n_nodes = curr_tree.list_node.size();
  vector<node> new_nodes;

  // Can't prune a root
  if(curr_tree.list_node.size()==1){
    *id_t = 1;
    return curr_tree;
  }

  // Getting the parents of terminal nodes
  vector<node> parent_left_right;
  for(int i=0;i<n_nodes;i++){
    if(!curr_tree.list_node[i].isTerminal()){
      if(curr_tree.list_node[curr_tree.list_node[i].left].isTerminal()){
        if(curr_tree.list_node[curr_tree.list_node[i].right].isTerminal()){
          parent_left_right.push_back(curr_tree.list_node[i]); // Adding the parent
        }
      }
    }
  }


  // Getting the number of internal and the node to be pruned
  int n_parent = parent_left_right.size();
  int p_node = sample_int(n_parent);


  // Returning to be a terminal node
  int left_index = curr_tree.list_node[parent_left_right[p_node].index].left;
  int right_index = curr_tree.list_node[parent_left_right[p_node].index].right;
  curr_tree.list_node[parent_left_right[p_node].index].left=-1;
  curr_tree.list_node[parent_left_right[p_node].index].right=-1;

  // Storing the index of the node that was pruned
  *verb_node_index = parent_left_right[p_node].index;

  // Adding new trees

  for(int k = 0;k<curr_tree.list_node.size();k++){
    if((curr_tree.list_node[k].index != left_index) && (curr_tree.list_node[k].index != right_index)){


      // Doing for trees greater than left index
      if(curr_tree.list_node[k].index>left_index){ /// COULD ALSO USE IF curr_tree.list_node[k].depth >depth_prune_node
        curr_tree.list_node[k].index = curr_tree.list_node[k].index-2;
      }
      // Checking if the left index is greater than it should be
      if(curr_tree.list_node[k].left>left_index){
        curr_tree.list_node[k].left = curr_tree.list_node[k].left-2;
      }
      // Checking if the right index is greater than it should be
      if(curr_tree.list_node[k].right>left_index){
        curr_tree.list_node[k].right = curr_tree.list_node[k].right-2;
      }

      new_nodes.push_back(curr_tree.list_node[k]);
    }
  }


  // Updating the new nodes
  curr_tree.list_node = new_nodes;
  return curr_tree;
}

// Change a tree
// [[Rcpp::depends(RcppEigen)]]
Tree change_bart(Tree curr_tree,
            Eigen::MatrixXd x_train,
            Eigen::MatrixXd x_test,
            int n_min_size,
            int* id_t,
            int* verb_node_index){

  // Declaring important values and variables
  int n_nodes = curr_tree.list_node.size();
  int p = x_train.cols(),n_train,n_test;
  int c_var,c_node;
  double c_rule;
  int bad_tree_counter=0;
  bool bad_tree = true;

  // Getting observations that are on the left and the ones that are in the right
  vector<int> new_left_train_index;
  vector<int> new_right_train_index;
  vector<int> curr_obs_train; // Observations that belong to that terminal node

  // Getting observations that are on the left and the ones that are in the right
  vector<int> new_left_test_index;
  vector<int> new_right_test_index;
  vector<int> curr_obs_test; // Observations that belong to that terminal node


  // Can't prune a root
  if(curr_tree.list_node.size()==1){
    *id_t = 1;
    return curr_tree;
  }

  // Getting the parents of terminal nodes (NOG)
  vector<node> parent_left_right;
  for(int i=0;i<n_nodes;i++){
    if(curr_tree.list_node[i].isTerminal()==0){
      if(curr_tree.list_node[curr_tree.list_node[i].left].isTerminal()==1 && curr_tree.list_node[curr_tree.list_node[i].right].isTerminal()==1){
        parent_left_right.push_back(curr_tree.list_node[i]); // Adding the parent
      }
    }
  }


  // Getting the number of parent of terminals and selecting the one to be changed
  int n_parent = parent_left_right.size();

  while(bad_tree){

    // Selecting the node and var
    c_node = sample_int(n_parent);
    c_var = sample_int(p);
    c_rule = sample_double(x_train.col(c_var),n_min_size);

    // Number of train observations that will be changed
    curr_obs_train = parent_left_right[c_node].obs_train;
    n_train = parent_left_right[c_node].obs_train.size();

    // Number of test observations that will be changed
    curr_obs_test = parent_left_right[c_node].obs_test;
    n_test = parent_left_right[c_node].obs_test.size();

    // Iterating over the train observations
    for(int i = 0; i<n_train; i++){
      if(x_train.coeff(curr_obs_train[i],c_var)<=c_rule){
        new_left_train_index.push_back(curr_obs_train[i]);
      } else {
        new_right_train_index.push_back(curr_obs_train[i]);
      }
    }

    // Iterating over the test observations
    for(int i = 0; i<n_test; i++){
      if(x_test.coeff(curr_obs_test[i],c_var)<=c_rule){
        new_left_test_index.push_back(curr_obs_test[i]);
      } else {
        new_right_test_index.push_back(curr_obs_test[i]);
      }
    }


    // Verifying if is the correct min node size
    if(new_right_train_index.size()>=n_min_size && new_left_train_index.size()>=n_min_size){

      // Returning the nodes that will be changed
      int left_index = curr_tree.list_node[parent_left_right[c_node].index].left;
      int right_index = curr_tree.list_node[parent_left_right[c_node].index].right;


      // Changing the rules and observations that belong to each one of
      // that nodes
      curr_tree.list_node[left_index].var = c_var;
      curr_tree.list_node[right_index].var = c_var;
      curr_tree.list_node[left_index].var_split = c_rule;
      curr_tree.list_node[right_index].var_split = c_rule;
      curr_tree.list_node[left_index].obs_train = new_left_train_index;
      curr_tree.list_node[right_index].obs_train = new_right_train_index;


      // Changing the test observations
      curr_tree.list_node[left_index].obs_test = new_left_test_index;
      curr_tree.list_node[right_index].obs_test = new_right_test_index;

      // storing the index of the changed node
      *verb_node_index = parent_left_right[c_node].index;
      // Success to create a good tree
      bad_tree = false;

    } else {

      // Cleaning the current vectors
      new_left_train_index.clear();
      new_right_train_index.clear();
      curr_obs_train.clear();

      new_left_test_index.clear();
      new_right_test_index.clear();
      curr_obs_test.clear();


      bad_tree_counter++;

      // Avoid infinite while
      if(bad_tree_counter==2){
        *id_t = 1;
        bad_tree = false;
      }

    }

  }

  return curr_tree;

};


// [[Rcpp::depends(RcppEigen)]]
double node_loglikelihood_bart(Eigen::VectorXd residuals,
                          node current_node,
                          double tau,
                          double tau_mu) {

  // Decarling the quantities
  int n_size = current_node.obs_train.size();
  double sum_sq_r = 0 , sum_r = 0;

  for(int i = 0;i<n_size;i++){
    sum_sq_r+=residuals.coeff(i)*residuals.coeff(i);
    sum_r+=residuals.coeff(i);

  }

  return -0.5*tau*sum_sq_r-0.5*tau*tau*sum_r*sum_r/(tau*n_size+tau_mu)-0.5*log(tau*n_size+tau_mu);
}

// [[Rcpp::depends(RcppEigen)]]
double tree_loglikelihood_bart(Eigen::VectorXd residuals,
                          Tree curr_tree,
                          double tau,
                          double tau_mu
) {

  // Declaring important quantities
  vector<node> terminal_nodes = curr_tree.getTerminals();
  int number_nodes = terminal_nodes.size();
  double loglike_sum = 0;


  for(int i = 0; i<number_nodes; i++) {
    loglike_sum+=node_loglikelihood_bart(residuals,
                                    terminal_nodes[i],tau,tau_mu);
  }

  return loglike_sum;
}

// [[Rcpp::depends(RcppEigen)]]
double tree_loglikelihood_verb(Eigen::VectorXd residuals,
                               Tree new_tree,
                               Tree curr_tree,
                               double verb,
                               int verb_node_index,
                               double tau,
                               double tau_mu){
  if(verb < 0.28){ // Grow prob
    return (node_loglikelihood_bart(residuals,new_tree.list_node[new_tree.list_node[verb_node_index].left],tau,tau_mu)+node_loglikelihood_bart(residuals,new_tree.list_node[new_tree.list_node[verb_node_index].right],tau,tau_mu)) - node_loglikelihood_bart(residuals,curr_tree.list_node[verb_node_index],tau,tau_mu);
  } else if( verb <= 0.56){ // Prune prob
    return node_loglikelihood_bart(residuals, curr_tree.list_node[verb_node_index],tau,tau_mu)-(node_loglikelihood_bart(residuals, curr_tree.list_node[curr_tree.list_node[verb_node_index].left],tau,tau_mu)+node_loglikelihood_bart(residuals, curr_tree.list_node[curr_tree.list_node[verb_node_index].right],tau,tau_mu));
  } else { // Change case
    return (node_loglikelihood_bart(residuals,new_tree.list_node[new_tree.list_node[verb_node_index].left],tau,tau_mu)+node_loglikelihood_bart(residuals,new_tree.list_node[new_tree.list_node[verb_node_index].right],tau,tau_mu))-(node_loglikelihood_bart(residuals, curr_tree.list_node[curr_tree.list_node[verb_node_index].left],tau,tau_mu)+node_loglikelihood_bart(residuals, curr_tree.list_node[curr_tree.list_node[verb_node_index].right],tau,tau_mu));
  }

}



// Updating the \mu
// [[Rcpp::depends(RcppEigen)]]
Tree update_mu(Eigen::VectorXd residuals,
               Tree curr_tree,
               double tau,
               double tau_mu){

  // Declaring important quantities
  vector<node> terminal_nodes = curr_tree.getTerminals();
  int number_nodes = terminal_nodes.size();
  double sum_residuals;

  // Iterating over terminal nodes
  for(int i = 0;i<number_nodes;i++){
    sum_residuals = 0;
    for(int j = 0;j<terminal_nodes[i].obs_train.size();j++){
      sum_residuals+=residuals.coeff(terminal_nodes[i].obs_train[j]);
    }
    curr_tree.list_node[terminal_nodes[i].index].mu = R::rnorm( ((tau)/(terminal_nodes[i].obs_train.size()*tau+tau_mu))*sum_residuals,sqrt(1/(terminal_nodes[i].obs_train.size()*tau+tau_mu)) );

  }

  return curr_tree;
}

// Calculate the density of a half cauchy location 0 and sigma
//[[Rcpp::export]]
double dhcauchy(double x, double location, double sigma){

  if( x>location) {
    return (1/(M_PI_2*sigma))*(1/(1+((x-location)*(x-location))/(sigma*sigma)));
  } else {
    return 0.0;
  }
}
//
// double update_tau_old(Eigen::VectorXd y,
//                   Eigen::VectorXd y_hat,
//                   double a_tau,
//                   double d_tau){
//
//   // Function used in the development of the package where I checked
//   // contain_nan(y_hat);
//
//   int n = y.size();
//   double sum_sq_res=0;
//   for(int i=0;i<n; i++){
//       sum_sq_res += (y.coeff(i)-y_hat.coeff(i))*(y.coeff(i)-y_hat.coeff(i));
//   }
//   return R::rgamma((0.5*n+a_tau),1/(0.5*sum_sq_res+d_tau));
// }
//
double update_tau(Eigen::VectorXd y,
                  Eigen::VectorXd y_hat,
                  double naive_sigma,
                  double curr_tau){

  int n=y.size();
  double curr_sigma, proposal_tau,proposal_sigma, acceptance;

  curr_sigma = 1/(sqrt(curr_tau));

  // Getting the sum squared of residuals
  double sum_sq_res=0;
  for(int i=0;i<n; i++){
    sum_sq_res += ((y.coeff(i)-y_hat.coeff(i)))*((y.coeff(i)-y_hat.coeff(i)));
  }

  // Getting a proposal sigma
  proposal_tau = R::rgamma(0.5*n+1,1/(0.5*sum_sq_res));
  proposal_sigma  = 1/(sqrt(proposal_tau));

  acceptance = exp(log(dhcauchy(proposal_sigma,0,naive_sigma))+3*log(proposal_sigma)-log(dhcauchy(curr_sigma,0,naive_sigma))-3*log(curr_sigma));

  if(R::runif(0,1)<=acceptance){
    return 1/(proposal_sigma*proposal_sigma);
  } else{
    return curr_tau;
  }
}
//
//
//
//
//
// [[Rcpp::depends(RcppEigen)]]
void get_prediction_tree_bart(Tree curr_tree,
                         Eigen::MatrixXd x_train,
                         Eigen::MatrixXd x_test,
                         Eigen::VectorXd &prediction_train,
                         Eigen::VectorXd &prediction_test){



  // Getting terminal nodes
  vector<node> terminal_nodes = curr_tree.getTerminals();
  int n_terminals = terminal_nodes.size();


  vector<int> curr_obs_train;
  vector<int> new_left_train_index, new_right_test_index;

  // Iterating to get the predictions for a training test
  for(int i = 0;i<n_terminals;i++){

    // Updating the training predictions
    for(int j = 0; j<terminal_nodes[i].obs_train.size();j++){
      prediction_train[terminal_nodes[i].obs_train[j]] = terminal_nodes[i].mu;
    }

    for(int j = 0; j<terminal_nodes[i].obs_test.size();j++){
      prediction_test[terminal_nodes[i].obs_test[j]] = terminal_nodes[i].mu;
    }

  }

}

// [[Rcpp::depends(RcppEigen)]]
double tree_log_prior_verb(Tree new_tree,
                           Tree curr_tree,
                           double verb,
                           double alpha,
                           double beta) {

  double log_tree_p = 0.0;

  if (verb<0.56){

    // Adding for new tree
    for(int i=0;i<curr_tree.list_node.size();i++){
      if(curr_tree.list_node[i].isTerminal()==1){
        log_tree_p -= log(1-alpha/pow((1+curr_tree.list_node[i].depth),beta));
      } else {
        log_tree_p -= log(alpha)-beta*log(1+curr_tree.list_node[i].depth);
      }
    }

    // Subtracting for the current tree
    for(int i=0;i<new_tree.list_node.size();i++){
      if(new_tree.list_node[i].isTerminal()==1){
        log_tree_p -= log(1-alpha/pow((1+new_tree.list_node[i].depth),beta));
      } else {
        log_tree_p -= log(alpha)-beta*log(1+new_tree.list_node[i].depth);
      }
    }

  }

  return log_tree_p;
}

// Get the log transition probability
// [[Rcpp::depends(RcppEigen)]]
double log_transition_prob(Tree curr_tree,
                           Tree new_tree,
                           double verb){

  // Getting the probability
  double log_prob = 0;
  // In case of Grow: (Prob from Grew to Current)/(Current to Grow)
  if(verb < 0.3){
    log_prob = log(0.3/new_tree.n_nog())-log(0.3/curr_tree.n_terminal());
  } else if (verb < 0.6) { // In case of Prune: (Prob from the Pruned to the current)/(Prob to the current to the prune)
    log_prob = log(0.3/new_tree.n_terminal())-log(0.3/curr_tree.n_nog());
  }; // In case of change log_prob = 0; it's already the actual value

  return log_prob;

}


//[[Rcpp::export]]
gpb_Tree convert_tree


//======== GP-BART FUNCTIONS============
//[[Rcpp::export]]
List gpbart(Eigen::MatrixXd x_train,
            Eigen::VectorXd y,
            Eigen::MatrixXd x_test,
            int n_tree,
            int n_mcmc,
            int n_burn,
            int n_min_size,
            int n_bart_iter,
            double tau, double mu,
            double tau_mu, double naive_sigma,
            double alpha, double beta){

  // Defining the parameters from GP-BART
  // GP-BART functions
  vector<double> phi_vec;
  double phi_init;
  double nu;
  double kappa;

  // Declaring common variables
  double verb;
  double acceptance;
  double log_transition_prob_obj;
  double past_tau;
  int post_counter = 0;
  int id_t ,verb_node_index; // Id_t: Boolean to verify if the same tree was generated
  bool bart_bool = true;
  // Verb_node_index: Explicit the node the was used;

  // Getting the number of observations
  int n_train = x_train.rows();
  int n_test = x_test.rows();

  // Creating the variables
  int n_post = (n_mcmc-n_burn);
  Eigen::MatrixXd y_train_hat_post(n_post,n_train);
  Eigen::MatrixXd y_test_hat_post(n_post,n_train);

  NumericVector tau_post;

  // Creating the initial values for omega, omega_plus_tau, and omega_plus_tau_inv
  // Defining initial values for omega matrices
  Eigen::MatrixXd i_omega,i_omega_plus_tau,i_omega_plus_tau_inv;


  // Defining the matrices for the root node
  i_omega = omega(x_train,phi_init,nu);
  i_omega_plus_tau = omega_plus_tau(x_train,phi_init,nu,tau);
  i_omega_plus_tau_inv = M_solve(i_omega_plus_tau);

  // Initialize the new tree
  Tree new_tree_bart(n_train,n_test);
  gpb_Tree new_tree(n_train,n_test,i_omega,i_omega_plus_tau,i_omega_plus_tau_inv);

  // Creating a vector of multiple trees
  vector<Tree> current_trees_bart;
  vector<gpb_Tree> current_trees;
  for(int i = 0; i<n_tree;i++){
    current_trees_bart.push_back(new_tree_bart);
  }


  // Creating a matrix of zeros of y_hat
  y_train_hat_post.setZero(n_post,n_train);
  y_test_hat_post.setZero(n_post,n_test);

  // Creating the partial residuals and partial predictions
  Eigen::VectorXd partial_pred,partial_residuals;
  Eigen::VectorXd prediction_train,prediction_test; // Creating the vector only one prediction
  Eigen::VectorXd prediction_test_sum;

  // Initializing zero values
  partial_pred.setZero(n_train);
  partial_residuals.setZero(n_train);
  prediction_train.setZero(n_train);
  prediction_test.setZero(n_test);


  // Iterating over all MCMC samples
  for(int i=0;i<n_mcmc;i++){

    prediction_test_sum.setZero(n_test);

    if( i < n_bart_iter ){

      // Using the new bart_bool
      bart_bool = false;

      // Converting the new trees
      for(int j = 0; j<n_tree;j++){

        current_trees[j] ;

      }
    }

    if(bart_bool){

        // Iterating over the trees
        for(int t = 0; t<n_tree;t++){

          // Initializing if store tree or not
          id_t = 0;
          // Updating the prediction tree and prediction test
          get_prediction_tree_bart(current_trees_bart[t],x_train,x_test,prediction_train,prediction_test);

          // Updating the residuals
          partial_residuals = y - (partial_pred - prediction_train);

          // Sampling a verb. In this case I will sample a double between 0 and 1
          // Grow: 0-0.3
          // Prune 0.3 - 0.6
          // Change: 0.6 - 1.0
          // Swap: Not in this current implementation
          verb = R::runif(0,1);

          // Forcing the first trees to grow
          if(current_trees[t].list_node.size()==1 ){
            verb = 0.1;
          }

          // Proposing a new tree
          if(verb<=0.28){
            new_tree_bart = grow_bart(current_trees_bart[t], x_train,x_test, n_min_size,&id_t,&verb_node_index);
          } else if ( verb <= 0.56){
            if(current_trees[t].list_node.size()>1){
              new_tree_bart = prune_bart(current_trees_bart[t],&id_t,&verb_node_index);
            }
          } else if (verb <= 1){
            new_tree_bart = change_bart(current_trees_bart[t],x_train,x_test,n_min_size,&id_t,&verb_node_index);
          } else {
            // new_tree = swap(current_trees[t],x_train,x_test,n_min_size);
          }

          // Calculating or not the likelihood (1 is for case where the trees are the same)
          if(id_t == 0){

            // No new tree is proposed (Jump log calculations)
            if( (verb <=0.6) && (current_trees_bart[t].list_node.size()==new_tree.list_node.size())){
              log_transition_prob_obj = 0;
            } else {
              log_transition_prob_obj = log_transition_prob(current_trees_bart[t],new_tree_bart,verb);

            }

            acceptance = tree_loglikelihood_verb(partial_residuals,new_tree_bart,current_trees_bart[t],verb,verb_node_index,tau, tau_mu) + tree_log_prior_verb(new_tree_bart,current_trees_bart[t],verb,alpha,beta)+ log_transition_prob_obj;

            // Testing if will acceptance or not
            if( (acceptance > 0) || (acceptance > -R::rexp(1))){
              current_trees[t] = new_tree;
            }

          } // End the test likelihood calculating for same trees

          // Generating new \mu values for the accepted (or not tree)
          current_trees_bart[t] = update_mu(partial_residuals,current_trees_bart[t],tau,tau_mu);

          // Updating the prediction train and prediction test with the sampled \mu values
          get_prediction_tree_bart(current_trees_bart[t],x_train,x_test,prediction_train,prediction_test);

          partial_pred = y + prediction_train - partial_residuals;
          prediction_test_sum += prediction_test;

        }

    } else { // Begin for the GP-BART iterations

    }
    // Updating tau
    past_tau = tau;
    tau = update_tau(y,partial_pred,naive_sigma,past_tau);
    // tau = update_tau_old(y,partial_pred,a_tau,d_tau);

    // Updating the posterior matrix
    if(i>=n_burn){
      y_train_hat_post.row(post_counter) += partial_pred;
      y_test_hat_post.row(post_counter) += prediction_test_sum;
      //Updating tau
      tau_post.push_back(tau);
      post_counter++;
    }

  }

  return Rcpp::List::create(_["y_train_hat_post"] = y_train_hat_post,
                            _["y_test_hat_post"] = y_test_hat_post,
                            _["tau_post"] = tau_post);

}





//[[Rcpp::export]]
List bart(Eigen::MatrixXd x_train,
          Eigen::VectorXd y,
          Eigen::MatrixXd x_test,
          int n_tree,
          int n_mcmc,
          int n_burn,
          int n_min_size,
          double tau, double mu,
          double tau_mu, double naive_sigma,
          double alpha, double beta){

  // Declaring common variables
  double verb;
  double acceptance;
  double log_transition_prob_obj;
  double past_tau;
  int post_counter = 0;
  int id_t ,verb_node_index; // Id_t: Boolean to verify if the same tree was generated
  // Verb_node_index: Explicit the node the was used;

  // Getting the number of observations
  int n_train = x_train.rows();
  int n_test = x_test.rows();

  // Creating the variables
  int n_post = (n_mcmc-n_burn);
  Eigen::MatrixXd y_train_hat_post(n_post,n_train);
  Eigen::MatrixXd y_test_hat_post(n_post,n_train);

  NumericVector tau_post;

  Tree new_tree(n_train,n_test);

  // Creating a vector of multiple trees
  vector<Tree> current_trees;
  for(int i = 0; i<n_tree;i++){
    current_trees.push_back(new_tree);
  }


  // Creating a matrix of zeros of y_hat
  y_train_hat_post.setZero(n_post,n_train);
  y_test_hat_post.setZero(n_post,n_test);

  // Creating the partial residuals and partial predictions
  Eigen::VectorXd partial_pred,partial_residuals;
  Eigen::VectorXd prediction_train,prediction_test; // Creating the vector only one prediction
  Eigen::VectorXd prediction_test_sum;

  // Initializing zero values
  partial_pred.setZero(n_train);
  partial_residuals.setZero(n_train);
  prediction_train.setZero(n_train);
  prediction_test.setZero(n_test);


  // Iterating over all MCMC samples
  for(int i=0;i<n_mcmc;i++){

    prediction_test_sum.setZero(n_test);


    // Iterating over the trees
    for(int t = 0; t<n_tree;t++){

      // Initializing if store tree or not
      id_t = 0;
      // Updating the prediction tree and prediction test
      get_prediction_tree(current_trees[t],x_train,x_test,prediction_train,prediction_test);

      // Updating the residuals
      partial_residuals = y - (partial_pred - prediction_train);

      // Sampling a verb. In this case I will sample a double between 0 and 1
      // Grow: 0-0.3
      // Prune 0.3 - 0.6
      // Change: 0.6 - 1.0
      // Swap: Not in this current implementation
      verb = R::runif(0,1);

      // Forcing the first trees to grow
      if(current_trees[t].list_node.size()==1 ){
        verb = 0.1;
      }

      // Proposing a new tree
      if(verb<=0.28){
        new_tree = grow(current_trees[t], x_train,x_test, n_min_size,&id_t,&verb_node_index);
      } else if ( verb <= 0.56){
        if(current_trees[t].list_node.size()>1){
          new_tree = prune(current_trees[t],&id_t,&verb_node_index);
        }
      } else if (verb <= 1){
        new_tree = change(current_trees[t],x_train,x_test,n_min_size,&id_t,&verb_node_index);
      } else {
        // new_tree = swap(current_trees[t],x_train,x_test,n_min_size);
      }

      // Calculating or not the likelihood (1 is for case where the trees are the same)
      if(id_t == 0){

        // No new tree is proposed (Jump log calculations)
        if( (verb <=0.6) && (current_trees[t].list_node.size()==new_tree.list_node.size())){
          log_transition_prob_obj = 0;
        } else {
          log_transition_prob_obj = log_transition_prob(current_trees[t],new_tree,verb);

        }

        acceptance = tree_loglikelihood_verb(partial_residuals,new_tree,current_trees[t],verb,verb_node_index,tau, tau_mu) + tree_log_prior_verb(new_tree,current_trees[t],verb,alpha,beta)+ log_transition_prob_obj;

        // Testing if will acceptance or not
        if( (acceptance > 0) || (acceptance > -R::rexp(1))){
          current_trees[t] = new_tree;
        }

      } // End the test likelihood calculating for same trees

      // Generating new \mu values for the accepted (or not tree)
      current_trees[t] = update_mu(partial_residuals,current_trees[t],tau,tau_mu);

      // Updating the prediction train and prediction test with the sampled \mu values
      get_prediction_tree(current_trees[t],x_train,x_test,prediction_train,prediction_test);

      partial_pred = y + prediction_train - partial_residuals;
      prediction_test_sum += prediction_test;
    }

    // Updating tau
    past_tau = tau;
    tau = update_tau(y,partial_pred,naive_sigma,past_tau);
    // tau = update_tau_old(y,partial_pred,a_tau,d_tau);

    // Updating the posterior matrix
    if(i>=n_burn){
      y_train_hat_post.row(post_counter) += partial_pred;
      y_test_hat_post.row(post_counter) += prediction_test_sum;
      //Updating tau
      tau_post.push_back(tau);
      post_counter++;
    }

  }

  return Rcpp::List::create(_["y_train_hat_post"] = y_train_hat_post,
                            _["y_test_hat_post"] = y_test_hat_post,
                            _["tau_post"] = tau_post);

}


// //[[Rcpp::export]]
// int initialize_test(Eigen::MatrixXd x_train,Eigen::MatrixXd x_test,
//                     Eigen::VectorXd residuals,
//                     double tau, double tau_mu, int node_min_size){
//
//   int id_t = 0;
//   int verb_node_index = -1;
//   int n_obs_train(x_train.rows()), n_obs_test(x_test.rows());
//   Tree tree1(n_obs_train,n_obs_test);
//   Tree new_tree = grow(tree1,x_train,x_test,node_min_size, &id_t,&verb_node_index);
//   id_t = 0;
//   verb_node_index = -1;
//   Tree new_tree_two = grow(new_tree,x_train,x_test,node_min_size, &id_t,&verb_node_index);
//   Tree new_tree_three = grow(new_tree_two,x_train,x_test,node_min_size, &id_t,&verb_node_index);
//   Tree new_tree_four = grow(new_tree_three,x_train,x_test,node_min_size, &id_t,&verb_node_index);
//   Tree new_tree_five = grow(new_tree_four,x_train,x_test,node_min_size, &id_t,&verb_node_index);
//   Tree new_tree_six = grow(new_tree_five,x_train,x_test,node_min_size, &id_t,&verb_node_index);
//
//
//   id_t = 0;
//   verb_node_index = -1;
//
//   new_tree_six.DisplayNodes();
//   cout << "======= SWAPP LINE =======" << endl;
//
//   Tree swap_tree = swap_old(new_tree_six,x_train,node_min_size);
//   // Tree swap_tree = prune(new_tree_five, &id_t,&verb_node_index);
//
//   swap_tree.DisplayNodes();
//   cout << " ==== " << endl;
//   cout << " Same Tree was generated: "<<id_t << endl;
//   cout << " ==== " << endl;
//
//   cout << " ==== " << endl;
//   cout << " The modified node was: "<< verb_node_index << endl;
//   cout << " ==== " << endl;
//
//
//   // Tree new_tree_three = grow(new_tree_two,x,2);
//   // Tree new_tree_four = grow(new_tree_three,x,2);
//   // Tree new_tree_five = grow(new_tree_four,x,2);
//   // Tree new_tree_six = grow(new_tree_five,x,2);
//   // new_tree_four.DisplayNodes();
//   // // Tree new_three_tree = grow(new_tree_two,x,2);
//
//   // cout << "======= SWAPP LINE =======" << endl;
//   // Tree swap_tree(n_obs);
//   // get_prediction_tree(new_tree_five,x,true);
//   // for(int k = 0;k<1000;k++){
//     // cout << k << endl;
//     // swap_tree = swap(new_tree_four,x,2);
//     // swap_tree.DisplayNodes();
//   // }
//   // Tree update_mu_tree = update_mu(residuals,change_tree_two,tau,tau_mu);
//   // Eigen::VectorXd pred_vec  = get_prediction_tree(update_mu_tree,x_new,false);
//   // // get_prediction_tree(change_tree_two,x,true);
//   // cout << "Pred x: ";
//   // for(int i = 0; i<pred_vec.size();i++){
//   //   cout  << pred_vec.coeff(i)<< " " ;
//   // }
//
//   // cout << "Tree prior" << tree_log_prior(new_tree_two,0.95,2) << endl;
//   // vector<int> p_index = new_tree_two.getParentTerminals();
//
//   // Tree prune_tree = prune(new_three_tree);
//   // prune_tree.DisplayNodes();
//
//   return 0;
//
// }





