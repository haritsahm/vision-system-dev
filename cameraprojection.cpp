#include "cameraprojection.h"

using namespace cv;
using namespace std;

CameraProjection::CameraProjection(Eigen::Matrix3d camInt, Eigen::VectorXd distVal, Eigen::MatrixXd camProj)
{
    this->K = camInt;
    if(distVal.size() < 5) std::cout << "Warning: Please fill with 5 distortion parameters" << std::endl;
    else
        this->D = distVal;
    if(camProj.rows() !=3 || camProj.cols() !=4) std::cout << "Warning: Please fill with Camera Projection 3x4" << std::endl;
    else
        this->P = camProj;

    k1 = D(0);
    k2 = D(1);
    p1 = D(2);
    p2 = D(3);
    k3 = D(4);

    fx = K(0,0);
    fy = K(1,1);
    cx = K(0,2);
    cy = K(1,2);

    fx_p= P(0,0);
    fy_p = P(1,1);
    cx_p = P(0,2);
    cy_p = P(1,2);
    Tx_p = P(0,3);
    Ty_p = P(1,3);

}


void CameraProjection::setCamIntrinsic(Eigen::MatrixXd camInt)
{
    this->K = camInt;

    fx = K(0,0);
    fy = K(1,1);
    cx = K(0,2);
    cy = K(1,2);

}

void CameraProjection::setDistValue(Eigen::VectorXd distVal)
{

    if(distVal.size() < 5) std::cout << "Warning: Please fill with 5 distortion parameters" << std::endl;
    else
    {
        this->D = distVal;
        k1 = D(0);
        k2 = D(1);
        p1 = D(2);
        p2 = D(3);
        k3 = D(4);
    }


}

void CameraProjection::setCamProj(Eigen::MatrixXd camProj)
{
    if(camProj.rows() !=3 || camProj.cols() !=4) std::cout << "Warning: Please fill with Camera Projection 3x4" << std::endl;
    else
    {
        this->P = camProj;

        fx_p= P(0,0);
        fy_p = P(1,1);
        cx_p = P(0,2);
        cy_p = P(1,2);
        Tx_p = P(0,3);
        Ty_p = P(1,3);
    }

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
