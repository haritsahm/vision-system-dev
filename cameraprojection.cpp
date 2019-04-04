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

void CameraProjection::undinstortPoint(Point2d &in, Point2d &res)
{

  double x = (in.x - cx) / fx;
  double y = (in.y - cy) / fy;

  double r2 = x*x + y*y;

  // Radial distorsion
  double xDistort = x * (1 + (k1*r2 + k2*r2 + k3*r2*r2)*r2);
  double yDistort = y * (1 + (k1*r2 + k2*r2 + k3*r2*r2)*r2);

  // Tangential distorsion
  xDistort = xDistort + (2*p1*x*y) + p2*(r2+2*x*x);
  yDistort = yDistort + p1*(r2+2*y*y) + (2*p2*x*y);

//  // Back to absolute coordinates.
  xDistort = xDistort * fx + cx;
  yDistort = yDistort * fy + cy;

  // Back to absolute coordinates.
  res.x = in.x * fx + cx;
  res.y = in.y * fy + cy;

}

void CameraProjection::undinstortPoints(std::vector<Point2d> &in, std::vector<Point2d> &res, cv::Size size)
{

  std::vector<Point2d> temp;
  undistortPoints(in, temp, K, D);
  for(auto point_: temp)
  {
    // To relative coordinates <- this is the step you are missing.
//    double x = (point_.x - cx) / fx;
//    double y = (point_.y - cy) / fy;

//    double centerx = (size.width/2 - cx)/fx;
//    double centery = (size.height/2 - cy)/fy;

//    double diff_x = x-centerx;
//    double diff_y = y-centery;

//    double r2 = diff_x*diff_x+diff_y*diff_y;
//    double r4 = r2*r2;
//    double r6 = r4*r2;

//    // Radial distorsion
//    double pixelDistort = (1 + k1*r2 + k2*r4 + k3*r6);

//    // Tangential distorsion
//    double tanDistX = (2*p1*x*y) + p2*(r2+2*x*x);
//    double tanDistY = p1*(r2+2*y*y) + (2*p2*x*y);

//    // Back to absolute coordinates.
//    double xCorr = x*pixelDistort+tanDistX;
//    double yCorr = y*pixelDistort+tanDistY;

    double u = point_.x*fx+cx;
    double v = point_.y*fy+cy;

    res.push_back(Point2d(u, v));
  }

}

void CameraProjection::convertToIPM(cv::Mat &in, cv::Mat &out, double &tilt)
{
  Mat R = (Mat_<double>(4,4) << 1, 0, 0, 0,
           0, cos(tilt*DEGREE2RADIAN), -sin(tilt*DEGREE2RADIAN), 0,
           0, sin(tilt*DEGREE2RADIAN), cos(tilt*DEGREE2RADIAN), 0,
           0, 0, 0, 1);

}

void CameraProjection::convertToIPM(Mat image, std::vector<Point2d> &points_in,
                                    std::vector<Point2d> &points_out, Eigen::Matrix3d &rot,
                                    Eigen::Vector3d &trans)
{

}
