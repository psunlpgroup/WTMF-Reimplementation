/* This is c++ script for ORMF re-implemetation 
   The raw code is in: ormf.m, written in matlab, located under weiwei/version-14-10-17/ormf. 
Reimplementation: with armadillo and OpenMP  
Brent: read_matrix, Initialize_PQ, build_index, 
Yanjun: matmul, compute_P, compute_Q, main 
w_m = 0.01, K=100 lambda = 20, alpha = 0.0001 iteration = 20 
*/ 

#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <assert.h>
#include <math.h>
// #include "omp.h"
#include <armadillo> 
#include <time.h>
#include <sys/time.h>

// #define THREAD_NUM 20 
// #define w_m 0.01
// #define K 100
// #define lambda 20
// #define alpha 0.0001
// #define iteration 20 
// #define maxiter 20 
// #define n_dim 100 

using namespace std; 
using namespace arma;


typedef long COORD;
typedef double VALUE;
typedef arma::sp_dmat SPARSE_MAT;
typedef arma::dmat DENSE_MAT;
typedef arma::dcolvec DENSE_COL;
// typedef arma::eye EYE_MAT; 

const char* filename = "train.ind";
const char* data_file = "output.txt";

const double w_m = 0.01;
const int K = 100;
const int lambda = 20;
const double alpha = 0.0001;
const int iteration = 20;
const int maxiter = 20;
const int n_dim = 100;


struct MatrixPair {
    // Represents two matrices
    DENSE_MAT p;
    DENSE_MAT q;
};

struct Index {
    // Represents two lists:
    // i: an index number
    // -- e.g., word index if indexing a document,
    //    document index if indexing a word
    // v: value at that index
    vector<COORD> i;
    vector<VALUE> v;
};

struct IndexPair {
    // Represents a pair of indices
    vector<Index> i4w;
    vector<Index> i4d;
};

// Helper: read time; 
double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

/* Read preprocessed matrix from files 
   Return a matrix */
SPARSE_MAT read_matrix(const char *filename, COORD &n_words, COORD &n_docs) {
    COORD rows = 0;
    COORD cols = 0;
    COORD word;
    COORD doc;
    VALUE score;
    ifstream fin;
    cout << "Analyzing file... ";
    cout.flush();
    fin.open(filename);
    while (fin >> word >> doc >> score) {
        if (word > rows) {
            rows = word;
        }
        if (doc > cols) {
            cols = doc;
        }
    }
    fin.close();
    cout << "Done!" << "\nWords: " << rows << ", Docs: " << cols << endl;
    SPARSE_MAT mat = SPARSE_MAT(rows, cols);
    n_words = rows; 
    n_docs = cols;

    cout << "Building matrix... ";
    cout.flush();
    fin.open(filename);
    while (fin >> word >> doc >> score) {
        word = word - 1;
        doc = doc - 1;
        mat(word, doc) = score;
    }
    fin.close();
    cout << "Done!" << endl;

    return mat; 
}

// Initialize matrix P and Q, where P is dim * n_words, Q is dim * n_docs 
// return matrix P, matrix Q 
// TODO: currently inits to the default Matlab random numbers
MatrixPair Initialize_PQ(COORD n_words, COORD n_docs) {
    MatrixPair pair;

    /* arma_rng::set_seed_random(); */

    /* pair.p = DENSE_MAT(n_dim, n_words); */
    /* pair.p.randn(); */

    DENSE_MAT P = DENSE_MAT(n_words, n_dim);

    mat::iterator it = P.begin();

    ifstream fin;
    fin.open("numbers.txt");
    VALUE num;
    while (fin >> num) {
        *it = num;
        ++it;
    }
    fin.close();

    pair.p = P.t();

    pair.q = DENSE_MAT(n_dim, n_docs);
    pair.q.zeros();

    return pair;
}

// Build index frmom matrix X 
// Return two list of pointers, correponding to cell array in matlab: i4d, i4w.  
IndexPair build_index(SPARSE_MAT X, COORD n_words, COORD n_docs) {
    vector<Index> i4w(n_words);
    vector<Index> i4d(n_docs);
    cout << "Building index... ";
    cout.flush();
    SPARSE_MAT::const_iterator it = X.begin();
    SPARSE_MAT::const_iterator it_end = X.end();
    for (COORD i, j; it != it_end; ++it) {
        i = it.row();
        j = it.col();
        i4w[i].i.push_back(j);
        i4w[i].v.push_back(*it);
        i4d[j].i.push_back(i);
        i4d[j].v.push_back(*it);
    }
    IndexPair pair;
    pair.i4w = i4w;
    pair.i4d = i4d;
    cout << "Done!" << endl;
    return pair;
}

// Return matrix P
DENSE_MAT compute_QP(MatrixPair matpair, IndexPair i4pair, int n_docs, int n_words){

    DENSE_MAT P = matpair.p; 
    DENSE_MAT Q = matpair.q; 

    DENSE_MAT EYE = eye(n_dim, n_dim);

    vector<Index> i4d = i4pair.i4d;
    vector<Index> i4w = i4pair.i4w;  

    // omp_set_num_threads(omp_get_num_procs());
    // cout << "Number of processors in use: " << omp_get_num_procs() << endl; 

    // #pragma omp parallel 
    for(int iter =0; iter < maxiter; iter++){
        cout << "WTMF training session : iteration = "<< iter << endl;
        cout << "WTMF training calculating p... " << endl;

        // Initialization;

        DENSE_MAT pptw = P * P.t() * w_m; 
        cout << "WTMF matrix Q " << endl;

        // Step 1 
        // Compute matrix Q
        // #pragma omp parallel for 
        for(COORD j=0; j<n_docs; j++){
            DENSE_MAT pv(n_dim,i4d[j].i.size()); 
            for (long unsigned int ii = 0; ii < i4d[j].i.size(); ++ii) { 
                pv.col(ii) = P.col(i4d[j].i[ii]);
            }
            // solve a system of linear equations 
            vec i4d_vec = vec(i4d[j].v);
            Q.col(j) = solve((pptw + pv*pv.t()*(1-w_m) + lambda*EYE), (pv*i4d_vec)); 
        }

        // Step 2
        // Compute matrix P
        DENSE_MAT qqtw = Q * Q.t() * w_m;    
        cout << "WTMF matrix P " << endl;
        // #pragma omp parallel for  
        for(COORD ind=0; ind<n_words; ind++){
            DENSE_MAT qv(n_dim,i4w[ind].i.size()); 
            for (long unsigned int ii = 0; ii < i4w[ind].i.size(); ++ii) {
                qv.col(ii) = Q.col(i4w[ind].i[ii]) ;
            }
            // solve a system of linear equations 
            vec i4w_vec = vec(i4w[ind].v);
            P.col(ind) = solve((qqtw + qv*qv.t()*(1-w_m) + lambda* EYE), (qv*i4w_vec));
        }

        // Orthognal projection 
        // #pragma omp critical 
        if(alpha!=0){
            cout << "WTMF gradient descent " << endl;
            // arma::trace(X) -- sum of the elements on the main diagonal of X
            // find the mean of the diagonal of P*P.t()
            double c = trace(P*P.t()) / n_dim;
            P = P - (alpha * ((P*P.t() - c*EYE) * P));
        }
    }

    return P;
}

//  Write matrix into mat file 
// void write_mat_data(char* data_file, Mat<double> &data_mat) {
void write_mat_data(const char* data_file, DENSE_MAT data_mat){
    ofstream out_file;

    cout << "[wtmf-corpus.cpp write_mat_data()]: writing " << data_file << endl;
    out_file.open(data_file);
    if (!out_file) {
        cout << endl << "[wtmf-corpus.cpp write_mat_data()]: Cannot open file " << data_file << endl;
        exit(1);
    }

    // Amardillo way to get rows and columns 
    cout << "[wtmf-corpus.cpp write_mat_data()]: cols=" << data_mat.n_cols << " rows=" << data_mat.n_rows << endl;
    out_file << data_mat.n_cols << " " << data_mat.n_rows << endl;
    for (uword i = 0; i < data_mat.n_cols; i++) {
        for (uword j = 0; j < data_mat.n_rows; j++) {
            out_file << data_mat(j,i) << " ";
        }
        out_file << endl;
    }
    out_file.close();
}

// function [P, Q] = ormf(X, dim, lambda, w_m, alpha, maxiter) 
// input is: filename for X, dim, lambda, w_m, alpha, max_iter 
// Output: generate matrix P and write into file 
int main( int argc, char **argv )
{	
    double simulation_time = read_timer();

    COORD n_words, n_docs; 
    SPARSE_MAT X = read_matrix(filename, n_words, n_docs); 

    MatrixPair matpair = Initialize_PQ(n_words, n_docs);  

    IndexPair i4pair = build_index(X, n_words, n_docs);

    DENSE_MAT P = compute_QP(matpair, i4pair, n_docs, n_words); 

    write_mat_data(data_file, P); 

    double end_time = read_timer(); 
    printf("Running for : %f time\n", end_time - simulation_time);

    return 0;
}

