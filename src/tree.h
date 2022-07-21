#include<iostream>
#include <vector>
#include "aux_functions.h"

using namespace std;


class node{

  // Storing parameters
public:
  int index; // Storing the node index
  vector<int> obs_train; // Storing train observations index that belong to that node
  vector<int> obs_test; // Storing test observations index that belong to that node
  int left; // Index of the left node
  int right; // Index of the right node
  int parent; // Index of the parent
  int depth; //  Depth of the node

  int var; // Variable which the node was split
  double var_split; // Variable which the node was split

  double mu; // Mu parameter from the node
  double loglike; // Loglikelihood of the residuals in that terminal node
public:

  // Getting the constructor
  node(int index_n, vector<int> obs_train_numb, vector<int> obs_test_numb,
       int left_i, int right_i, int depth_i, int var_i, double var_split_i,
       double mu_i){

    index = index_n;
    obs_train = obs_train_numb;
    obs_test = obs_test_numb;
    left = left_i;
    right = right_i;
    depth = depth_i;
    var = var_i;
    var_split = var_split_i;
    mu = mu_i;
  }

  void DisplayNode(){

    cout << "Node Number: " << index << endl;
    cout << "Decision Rule -> Var:  " << var << " & Rule: " << var_split << endl;
    cout << "Left <-  " << left << " & Right -> " << right << endl;

    if(true){
      cout << "Observations train: " ;
      for(int i = 0; i<obs_train.size(); i++){
        cout << obs_train[i] << " ";
      }
      cout << endl;
    }

    if(true){
      cout << "Observations test: " ;
      for(int i = 0; i<obs_test.size(); i++){
        cout << obs_test[i] << " ";
      }
      cout << endl;
    }

  }

  bool isTerminal(){
    return ((left == -1) && (right == -1) );
  }

};



class Tree{

  public:
    // Defining the main element of the tree structure
    vector<node> list_node;

    // Getting the vector of nodes
    Tree(int n_obs_train,int n_obs_test){
      // Creating a root node
      list_node.push_back(node(0,
                               seq_along(n_obs_train),
                               seq_along(n_obs_test),
                               -1, // left
                               -1, // right
                               0, //depth
                               -1, // var
                               -1.1, // var_split
                               0 )); // loglike
    }

    // void DisplayNodesNumber(){
    //   cout << "The tree has " << list_node.size() << " nodes" << endl;
    // }

    void DisplayNodes(){
      for(int i = 0; i<list_node.size(); i++){
        list_node[i].DisplayNode();
      }
      cout << "# ====== #" << endl;
    }


    // Getting terminal nodes
    vector<node> getTerminals(){

      // Defining terminal nodes
      vector<node> terminalNodes;

      for(int i = 0; i<list_node.size(); i++){
        if(list_node[i].isTerminal()==1){
          terminalNodes.push_back(list_node[i]); // Adding the terminals to the list
        }
      }
      return terminalNodes;
    }

    // Getting terminal nodes
    vector<node> getNonTerminals(){

      // Defining terminal nodes
      vector<node> NonTerminalNodes;

      for(int i = 0; i<list_node.size(); i++){
        if(list_node[i].isTerminal()==0){
          NonTerminalNodes.push_back(list_node[i]); // Adding the terminals to the list
        }
      }
      return NonTerminalNodes;
    }


    // Getting the number of n_terminals
    int n_terminal(){

      // Defining the sum value
      int terminal_sum = 0;
      for(int i = 0; i<list_node.size(); i++){
        if(list_node[i].isTerminal()==1){
          terminal_sum++;
        }
      }

      return terminal_sum;
    }

    // Getting the number of non-terminals
    int n_internal(){

      // Defining the sum value
      int internal_sum = 0;
      for(int i = 0; i<list_node.size(); i++){
        if(list_node[i].isTerminal()==0){
          internal_sum++;
        }
      }

      return internal_sum;
    }

    // Get the number of NOG (branches parents of terminal nodes)
    int n_nog(){
        int nog_counter = 0;
          for(int i=0;i<list_node.size();i++){
            if(list_node[i].isTerminal()==0){
              if(list_node[list_node[i].left].isTerminal()==1 && list_node[list_node[i].right].isTerminal()==1){
                nog_counter++;
              }
            }
        }
        return nog_counter;
      }
};

RCPP_EXPOSED_CLASS(node)
RCPP_EXPOSED_CLASS(Tree)

// GP BART classes and functions

class gpb_node{

  // Storing parameters
public:
  int index; // Storing the node index
  vector<int> obs_train; // Storing train observations index that belong to that node
  vector<int> obs_test; // Storing test observations index that belong to that node
  int left; // Index of the left node
  int right; // Index of the right node
  int parent; // Index of the parent
  int depth; //  Depth of the node

  int var; // Variable which the node was split
  double var_split; // Variable which the node was split

  double mu; // Mu parameter from the node
  Eigen::MatrixXd omega_plus_tau; // Getting the Omega_plus_tau
  Eigen::MatrixXd omega_plus_tau_inv;
  Eigen::MatrixXd omega;

public:

  // Getting the constructor
  gpb_node(int index_n, vector<int> obs_train_numb, vector<int> obs_test_numb,
           int left_i, int right_i, int depth_i, int var_i, double var_split_i,
           double mu_i,
           Eigen::MatrixXd omega_i,
           Eigen::MatrixXd omega_plus_tau_i,
           Eigen::MatrixXd omega_plus_tau_inv_i
           ){

    index = index_n;
    obs_train = obs_train_numb;
    obs_test = obs_test_numb;
    left = left_i;
    right = right_i;
    depth = depth_i;
    var = var_i;
    var_split = var_split_i;
    mu = mu_i;
    omega_plus_tau = omega_plus_tau_i;
    omega_plus_tau_inv = omega_plus_tau_inv_i;
    omega = omega_i;
  }

  void DisplayNode(){

    cout << "Node Number: " << index << endl;
    cout << "Decision Rule -> Var:  " << var << " & Rule: " << var_split << endl;
    cout << "Left <-  " << left << " & Right -> " << right << endl;

    if(true){
      cout << "Observations train: " ;
      for(int i = 0; i<obs_train.size(); i++){
        cout << obs_train[i] << " ";
      }
      cout << endl;
    }

    if(true){
      cout << "Observations test: " ;
      for(int i = 0; i<obs_test.size(); i++){
        cout << obs_test[i] << " ";
      }
      cout << endl;
    }

  }

  // Viewing a preview of this omega
  void prev_omega(){
    for(int i = 0; i< 5 ;i++){
      for(int j = 0 ; j< 5 ; j++){
        cout << " "<< omega(i,j) << " " ;
      }
      cout << endl;
    }
  }

  // Viewing \Omega + \digag{tau}
  void prev_omega_plus_tau(){
    for(int i = 0; i< 5 ;i++){
      for(int j = 0 ; j< 5 ; j++){
        cout << " "<< omega_plus_tau(i,j) << " " ;
      }
      cout << endl;
    }
  }

  // Viewing (\Omega + \digag{tau})^(-1)
  void prev_omega_plus_tau_inv(){
    for(int i = 0; i< 5 ;i++){
      for(int j = 0 ; j< 5 ; j++){
        cout << " "<< omega_plus_tau_inv(i,j) << " " ;
      }
      cout << endl;
    }
  }

  bool isTerminal(){
    return ((left == -1) && (right == -1) );
  }

};


// [[Rcpp::export]]
Eigen::MatrixXd omega(const Eigen::MatrixXd x_train,
                      const double phi,
                      const double nu){
  int nrow(x_train.rows());
  double kernel_input;
  Eigen::MatrixXd omega = Eigen::MatrixXd(nrow,nrow).setOnes();
  for(int i = 0; i < nrow; i++){
    omega(i,i) = 1/nu;
    for(int c = i+1; c<nrow; c++){
      kernel_input = exp(-((x_train.row(i)-x_train.row(c)).squaredNorm())/(2*phi*phi))/nu;// Calculate the squared omegaISTANCE matrix
      omega(i,c) =  kernel_input;
      omega(c,i) = kernel_input;
    }
  }
  return omega;
}

// [[Rcpp::export]]
Eigen::MatrixXd omega_plus_tau(const Eigen::MatrixXd x_train,
                               const double phi,
                               const double nu,
                               const double tau){
  int nrow(x_train.rows());
  double kernel_input;
  Eigen::MatrixXd omega = Eigen::MatrixXd(nrow,nrow).setOnes();
  for(int i = 0; i < nrow; i++){
    omega(i,i) = 1/nu + 1/tau;
    for(int c = i+1; c<nrow; c++){
      kernel_input = exp(-((x_train.row(i)-x_train.row(c)).squaredNorm())/(2*phi*phi))/nu;// Calculate the squared omegaISTANCE matrix
      omega(i,c) =  kernel_input;
      omega(c,i) = kernel_input;
    }
  }
  return omega;
}

// Creating a function to create the kernel matrix between A and B
// [[Rcpp::export]]
Eigen::MatrixXd distance_matrix(const Eigen::MatrixXd x_train,
                                const Eigen::MatrixXd x_test){
  int nrow1(x_train.rows());
  int nrow2(x_test.rows());
  int ncol1(x_train.cols());
  int ncol2(x_test.cols());

  if (ncol1 != ncol2) {
    throw std::runtime_error("Incompatible number of dimensions");
  }

  Eigen::MatrixXd D = Eigen::MatrixXd(nrow1,nrow2).setZero();
  for(int i = 0; i < nrow1; i++){
    for(int c = 0; c < nrow2; c++){
      D(i,c) = (x_train.row(i) - x_test.row(c)).squaredNorm();
    }
  }
  return D;
}



class gpb_Tree{

public:
  // Defining the main element of the tree structure
  vector<gpb_node> list_node;

  // Getting the vector of nodes
  gpb_Tree(int n_obs_train,int n_obs_test,
           Eigen::MatrixXd omega_matrix,
           Eigen::MatrixXd omega_matrix_plus_tau,
           Eigen::MatrixXd omega_matrix_plus_tau_inv){
    // Creating a root node
    list_node.push_back(gpb_node(0,
                                 seq_along(n_obs_train),
                                 seq_along(n_obs_test),
                                 -1, // left
                                 -1, // right
                                 0, //depth
                                 -1, // var
                                 -1.1, // var_split
                                 0, /// mu
                                 omega_matrix,
                                 omega_matrix_plus_tau,
                                 omega_matrix_plus_tau_inv));
  }

  // void DisplayNodesNumber(){
  //   cout << "The tree has " << list_node.size() << " nodes" << endl;
  // }

  void DisplayNodes(){
    for(int i = 0; i<list_node.size(); i++){
      list_node[i].DisplayNode();
    }
    cout << "# ====== #" << endl;
  }


  // Getting terminal nodes
  vector<gpb_node> getTerminals(){

    // Defining terminal nodes
    vector<gpb_node> terminalNodes;

    for(int i = 0; i<list_node.size(); i++){
      if(list_node[i].isTerminal()==1){
        terminalNodes.push_back(list_node[i]); // Adding the terminals to the list
      }
    }
    return terminalNodes;
  }

  // Getting terminal nodes
  vector<gpb_node> getNonTerminals(){

    // Defining terminal nodes
    vector<gpb_node> NonTerminalNodes;

    for(int i = 0; i<list_node.size(); i++){
      if(list_node[i].isTerminal()==0){
        NonTerminalNodes.push_back(list_node[i]); // Adding the terminals to the list
      }
    }
    return NonTerminalNodes;
  }


  // Getting the number of n_terminals
  int n_terminal(){

    // Defining the sum value
    int terminal_sum = 0;
    for(int i = 0; i<list_node.size(); i++){
      if(list_node[i].isTerminal()==1){
        terminal_sum++;
      }
    }

    return terminal_sum;
  }

  // Getting the number of non-terminals
  int n_internal(){

    // Defining the sum value
    int internal_sum = 0;
    for(int i = 0; i<list_node.size(); i++){
      if(list_node[i].isTerminal()==0){
        internal_sum++;
      }
    }

    return internal_sum;
  }

  // Get the number of NOG (branches parents of terminal nodes)
  int n_nog(){
    int nog_counter = 0;
    for(int i=0;i<list_node.size();i++){
      if(list_node[i].isTerminal()==0){
        if(list_node[list_node[i].left].isTerminal()==1 && list_node[list_node[i].right].isTerminal()==1){
          nog_counter++;
        }
      }
    }
    return nog_counter;
  }
};


RCPP_EXPOSED_CLASS(gpb_node)
RCPP_EXPOSED_CLASS(gpb_Tree)

