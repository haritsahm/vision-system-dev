#ifndef CAMERAPROJECTION_H
#define CAMERAPROJECTION_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <random>
#include <iostream>

#include <eigen3/Eigen/Eigen>

class CameraProjection
{
public:
    CameraProjection(Eigen::Matrix3d camInt, Eigen::VectorXd distVal, Eigen::MatrixXd camProj);

    void setCamIntrinsic(Eigen::MatrixXd camInt);
    void setDistValue(Eigen::VectorXd distVal);
    void setCamProj(Eigen::MatrixXd camProj);

    void undinstortPoint(cv::Point2d in, cv::Point2d &res);
    void undinstortPoints(std::vector<cv::Point2d> in, std::vector<cv::Point2d> &res);


    void convertToIPM(cv::Mat image, std::vector<cv::Point2d> points_in, std::vector<cv::Point2d> points_out,  Eigen::Matrix3d rot, Eigen::Vector3d trans);

private:
    Eigen::Matrix3d K;
    Eigen::VectorXd D;
    Eigen::Matrix<double, 3, 4> P;

    double k1,k2,k3,p1,p2;
    double fx,fy,cx,cy;
    double fx_p, fy_p, cx_p, cy_p, Tx_p, Ty_p;
};

#endif // CAMERAPROJECTION_H
