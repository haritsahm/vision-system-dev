#include "cameraprojection.h"

using namespace cv;
using namespace std;

CameraProjection::CameraProjection(cv::Mat camInt, cv::Mat distVal, cv::Mat camProj)
{
    this->K = camInt;
    this->D = distVal;

    this->P = camProj;

    k1 = D.at<double>(0,0);
    k2 = D.at<double>(0,1);
    p1 = D.at<double>(0,2);
    p2 = D.at<double>(0,3);
    k3 = D.at<double>(0,4);

    fx = K.at<double>(0,0);
    fy = K.at<double>(1,1);
    cx = K.at<double>(0,2);
    cy = K.at<double>(1,2);

    fx_p= P.at<double>(0,0);
    fy_p = P.at<double>(1,1);
    cx_p = P.at<double>(0,2);
    cy_p = P.at<double>(1,2);
    Tx_p = P.at<double>(0,3);
    Ty_p = P.at<double>(1,3);

}

CameraProjection::CameraProjection()
{
  K = Mat::ones(3,3,CV_64F);
  D = Mat::zeros(5,1,CV_64F);
  P = Mat::ones(3,4,CV_64F);
}


void CameraProjection::setCamIntrinsic(cv::Mat camInt)
{
    this->K = camInt;

    fx = K.at<double>(0,0);
    fy = K.at<double>(1,1);
    cx = K.at<double>(0,2);
    cy = K.at<double>(1,2);

}

void CameraProjection::setDistValue(cv::Mat distVal)
{


        this->D = distVal;
        k1 = D.at<double>(0,0);
        k2 = D.at<double>(0,1);
        p1 = D.at<double>(0,2);
        p2 = D.at<double>(0,3);
        k3 = D.at<double>(0,4);


}

void CameraProjection::setCamProj(cv::Mat camProj)
{
        this->P = camProj;
        fx_p= P.at<double>(0,0);
        fy_p = P.at<double>(1,1);
        cx_p = P.at<double>(0,2);
        cy_p = P.at<double>(1,2);
        Tx_p = P.at<double>(0,3);
        Ty_p = P.at<double>(1,3);
}

void CameraProjection::undinstortPoint(Point2d in, Point2d &res)
{

}

void CameraProjection::undinstortPoints(std::vector<Point2d> in, std::vector<Point2d> &res)
{

}

void CameraProjection::convertToIPM(Mat image, std::vector<Point2d> points_in,
                                    std::vector<Point2d> points_out, Eigen::Matrix3d rot,
                                    Eigen::Vector3d trans)
{

}
