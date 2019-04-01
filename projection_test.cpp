#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>

#include <yaml-cpp/yaml.h>
#include <random>
#include <iostream>
#include <math.h>
#include "cameraprojection.h"

using namespace cv;
using namespace std;

int main()
{

  Mat camera_matrix = (Mat_<double>(3,3) << 382.4136748158022, 0, 320.2835765599335, 0, 382.71400154214, 232.5968854679448, 0, 0, 1);
  Mat distortionCoefficients = (Mat_<double>(5,1) << -0.2231453634824506, 0.05371482506132921, -0.0008607002236214033, -0.0001954186291746453, 0);
  Mat camera_projection = (Mat_<double>(3,4) << 316.7410888671875, 0, 320.0782966205297, 0, 0, 346.2599792480469, 230.3749617875164, 0, 0, 0, 1, 0);

  std::cout << "Camera Matrix " << std::endl;
  std::cout << camera_matrix << std::endl;
  std::cout << "Distortion Matrix" << std::endl;
  std::cout << distortionCoefficients << std::endl;
  std::cout << "Projection Matrix" << std::endl;
  std::cout << camera_projection << std::endl;

  CameraProjection projection_(camera_matrix, distortionCoefficients, camera_projection);

  Mat image = imread("wide.png", IMREAD_COLOR);

  Mat out, matCam;

  vector<Point2d> dist_points, undist_points;
  dist_points.push_back(Point2d(284,31));
  dist_points.push_back(Point2d(404,50));
  dist_points.push_back(Point2d(548,58));
  dist_points.push_back(Point2d(257,140));
  dist_points.push_back(Point2d(453,140));
  dist_points.push_back(Point2d(578,89));

  for(auto point: dist_points)
  {
    circle(image, point, 2, Scalar(100,0,244));
  }

  undistort(image, out, camera_matrix, distortionCoefficients,matCam);

  imshow("original", image);
  imshow("undistrot", out);


  waitKey();


  return 0;
}
