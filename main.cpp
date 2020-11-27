#include <iostream>
#include "string.h"

#include <string>
#include <memory>
#include <Eigen/Dense>

using namespace std;

const Eigen::Matrix<double, 3, 3> computeTransformation(const Eigen::MatrixXd &source, 
                                                        const Eigen::MatrixXd &target, const int num_ref)
{   
    //Compute mean
    Eigen::Matrix<double, 2, 1> target_mean;
    for(int i = 0; i < target.cols(); i++)
    {
        target_mean(0, 0) += target(0, i);
        target_mean(1, 0) += target(1, i);
    }
    target_mean = target_mean/target.cols();
    // target_mean(0, 0) = (target(0, 0) + target(0, 1))/2;
    // target_mean(1, 0) = (target(1, 0) + target(1, 1))/2; 
    Eigen::Matrix<double, 2, 1> source_mean;
    for(int j = 0; j < source.cols(); j++)
    {
        source_mean(0, 0) += source(0, j);
        source_mean(1, 0) += source(1, j);
    }
    source_mean = source_mean/source.cols();
    // source_mean(0, 0) = (source(0, 0) + source(0, 1))/2;
    // source_mean(1, 0) = (source(1, 0) + source(1, 1))/2;

    //Compute different value
    Eigen::Matrix<double, 2, Eigen::Dynamic>source_difva(2, num_ref);
    for(int m = 0; m < source.cols(); m++)
    {
        source_difva(0, m) = source(0, m) -  source_mean(0,0);
        source_difva(1, m) = source(1, m) -  source_mean(1,0);
    }
    // source_difva(0, 0)= source(0, 0) - source_mean(0,0);
    // source_difva(0, 1)= source(0, 1) - source_mean(0,0);
    // source_difva(1, 0)= source(1, 0) - source_mean(1,0);
    // source_difva(1, 1)= source(1, 1) - source_mean(1,0);

    Eigen::Matrix<double, 2, Eigen::Dynamic>target_difva(2, num_ref);
    for(int n = 0; n < target.cols(); n++)
    {
        target_difva(0, n) = target(0, n) -  target_mean(0,0);
        target_difva(1, n) = target(1, n) -  target_mean(1,0);
    }
    // target_difva(0, 0)= target(0, 0) - target_mean(0,0);
    // target_difva(0, 1)= target(0, 1) - target_mean(0,0);
    // target_difva(1, 0)= target(1, 0) - target_mean(1,0);
    // target_difva(1, 1)= target(1, 1) - target_mean(1,0);

    Eigen::Matrix<double, 2, 2> H = source_difva * target_difva.transpose();
    // Compute the Singular Value Decomposition
    Eigen::JacobiSVD<Eigen::Matrix<double, 2, 2> > svd (H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 2, 2> u = svd.matrixU ();
    Eigen::Matrix<double, 2, 2> v = svd.matrixV ();
    if (u.determinant () * v.determinant () < 0)
    {
    for (int x = 0; x < 2; ++x)
      v (x, 2) *= -1;
    }
    
    Eigen::Matrix<double, 2, 2> R = v * u.transpose ();
    Eigen::Matrix<double, 2, 1> t = target_mean - R*source_mean; 
    Eigen::Matrix<double, 3, 3> transform_matrix;
    transform_matrix.setZero();
    transform_matrix.block<2, 2>(0, 0) = R;
    transform_matrix.block<2, 1>(0, 2) = t;
    transform_matrix(2 ,2) = 1;
    cout<<transform_matrix<<endl;
    return transform_matrix;
}


int main(int argc, char ** argv)
{
    int num_ref = 3;

    Eigen::MatrixXd source(2, num_ref);
    source << 2, 3, 4,
              1, 2, 3;
    Eigen::MatrixXd target(2, num_ref);
    target<<1.707, 1.707, 1.707,
            4.121, 5.535, 6.950;
    computeTransformation(source, target, num_ref);
    return 0;
}
