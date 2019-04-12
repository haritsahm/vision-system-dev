#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <eigen3/Eigen/Eigen>
#include <yaml-cpp/yaml.h>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include "linedetector.h"
//#include <linesegment.h>

#include <algorithm>
#include <random>
#include <math.h>

#define INF 1000000007

using namespace std;
using namespace cv;

string fieldBar = "Field Trackbar";
string BallBar = "Ball Trackbar";
string GoalBall = "Goal Trackbar";
string LineBar = "Line Trackbar";
string ObstBar = "Obstacle Trackbar";

void saveConfig(const string &path);
void loadConfig(const string &path);

Mat lut;
int bins = 64;
#define WHITE 1
#define GREEN 2
#define BLACK 3
#define OTHER 0

Mat image, RawHSVImg, RawBGRImg, RawYUVImg, undistImage;

struct Vec3 { int x; int y; int z; };
struct Vec6 {int x; int y; int z; int r; int p; int w;};

class hsvRangeC
{
public:
  bool active;
  int h0;
  int h1;
  int s0;
  int s1;
  int v0;
  int v1;

  void low(Eigen::Vector3d low){h0 = low.x(); s0 = low.y(); v0=low.z();}
  void high(Eigen::Vector3d high){h1 = high.x(); s1 = high.y(); v1=high.z();}
  void load(Vec6 data){h0 = data.x; h1 = data.y; s0 = data.z; s1 = data.r; v0 = data.p; v1 = data.w;}
};

hsvRangeC fieldRange, fieldYUV;
hsvRangeC lineRange, goalieHSV, goalieYUV;

namespace YAML {
template<>
struct convert<Eigen::Vector3d> {
  static Node encode(const Eigen::Vector3d& rhs) {
    Node node;
    node.push_back(rhs.x());
    node.push_back(rhs.y());
    node.push_back(rhs.z());
    return node;
  }

  static bool decode(const Node& node, Eigen::Vector3d& rhs) {
    if(!node.IsSequence() || node.size() != 3) {
      return false;
    }

    rhs.x() = node[0].as<double>();
    rhs.y() = node[1].as<double>();
    rhs.z() = node[2].as<double>();
    return true;
  }
};

template<>
struct convert<Vec6> {
  static Node encode(const Vec6& rhs) {
    Node node;
    node.push_back(rhs.x);
    node.push_back(rhs.y);
    node.push_back(rhs.z);
    node.push_back(rhs.r);
    node.push_back(rhs.p);
    node.push_back(rhs.w);
    return node;
  }

  static bool decode(const Node& node, Vec6& rhs) {
    if(!node.IsSequence() || node.size() != 6) {
      return false;
    }

    rhs.x = node[0].as<int>();
    rhs.y = node[1].as<int>();
    rhs.z = node[2].as<int>();
    rhs.r = node[3].as<int>();
    rhs.p = node[4].as<int>();
    rhs.w = node[5].as<int>();
    return true;
  }
};
}

YAML::Emitter& operator << (YAML::Emitter& out, Eigen::Vector3d& v) {
  out << YAML::Flow;
  out << YAML::BeginSeq << v.x() << v.y() << v.z() << YAML::EndSeq;
  return out;
}

YAML::Emitter& operator << (YAML::Emitter& out, Vec6& v) {
  out << YAML::Flow;
  out << YAML::BeginSeq << v.x << v.y << v.z << v.r << v.p << v.w << YAML::EndSeq;
  return out;
}

void getFieldLines(Mat& fieldbinary)
{
  //    Canny(fieldbinary, fieldbinary, 50, 200, 3); // Apply canny edge

  Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_ADV);

  double start = double(getTickCount());
  vector<Vec4f> lines_std;
  // Detect the lines
  ls->detect(fieldbinary, lines_std);
  double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
  std::cout << "It took " << duration_ms << " ms." << std::endl;
  // Show found lines
  Mat drawnLines(fieldbinary);
  ls->drawSegments(drawnLines, lines_std);
  imshow("Standard refinement", drawnLines);
}

void getGreenField(Mat& input, Mat& output)
{
  Mat channels[3];
  split(input, channels);

  Mat g = Mat::zeros(input.size(), CV_32FC1);

  for(int row = 0; row < input.rows; row++)
    for(int col = 0;col < input.cols; col++)
      g.at<float>(row,col) = (float)channels[1].at<uchar>(row,col) / ((float)channels[0].at<uchar>(row,col) + (float)channels[1].at<uchar>(row,col) + (float)channels[2].at<uchar>(row,col));

  g.convertTo(output, CV_8UC1, 255, 0);
}

void fieldTrackbar()
{
  namedWindow(fieldBar, CV_WINDOW_AUTOSIZE);
  createTrackbar("minH", fieldBar, &fieldRange.h0, 255);
  createTrackbar("minY", fieldBar, &fieldYUV.h0, 255);
  createTrackbar("maxY", fieldBar, &fieldYUV.h1, 255);
  createTrackbar("minU", fieldBar, &fieldYUV.s0, 255);
  createTrackbar("maxU", fieldBar, &fieldYUV.s1, 255);
  createTrackbar("minV", fieldBar, &fieldYUV.v0, 255);
  createTrackbar("maxV", fieldBar, &fieldYUV.v1, 255);
}

void lineTrackbar()
{
  namedWindow(LineBar, CV_WINDOW_AUTOSIZE);
  createTrackbar("minH", LineBar, &lineRange.h0, 255);
  createTrackbar("maxH", LineBar, &lineRange.h1, 255);
  createTrackbar("minS", LineBar, &lineRange.s0, 255);
  createTrackbar("maxS", LineBar, &lineRange.s1, 255);
  createTrackbar("minV", LineBar, &lineRange.v0, 255);
  createTrackbar("maxV", LineBar, &lineRange.v1, 255);
}

void goalieTrackbar()
{
  namedWindow(GoalBall, CV_WINDOW_AUTOSIZE);
  createTrackbar("minH", GoalBall, &goalieHSV.h0, 255);
  createTrackbar("maxH", GoalBall, &goalieHSV.h1, 255);
  createTrackbar("minS", GoalBall, &goalieHSV.s0, 255);
  createTrackbar("maxS", GoalBall, &goalieHSV.s1, 255);
  createTrackbar("minV", GoalBall, &goalieHSV.v0, 255);
  createTrackbar("maxV", GoalBall, &goalieHSV.v1, 255);
  createTrackbar("minY", GoalBall, &goalieYUV.h0, 255);
  createTrackbar("maxY", GoalBall, &goalieYUV.h1, 255);
  createTrackbar("minU", GoalBall, &goalieYUV.s0, 255);
  createTrackbar("maxU", GoalBall, &goalieYUV.s1, 255);
  createTrackbar("minVu", GoalBall, &goalieYUV.v0, 255);
  createTrackbar("maxVu", GoalBall, &goalieYUV.v1, 255);
}

vector<Point> contoursConvexHull( vector<vector<Point> > contours )
{
  vector<Point> result;
  vector<Point> pts;
  for ( size_t i = 0; i< contours.size(); i++)
    for ( size_t j = 0; j< contours[i].size(); j++)
    {
      double area = contourArea(contours[i]);

      if(area > 700)
        pts.push_back(contours[i][j]);
    }
  if(pts.size() > 0)
    convexHull( pts, result );
  return result;
}

vector<Point> locateFieldPoints(Mat &fieldBinary)
{
  vector<Point> fieldPoints;
  int STEP = 10;
  int maxCounter = 100;

  for(int col = 0; col < fieldBinary.cols; col+=STEP)
  {
    int blackCounter = 0;
    Point candidate;
    for(int row = fieldBinary.rows; row > 0; row--)
    {
      int pixel = fieldBinary.at<uchar>(row, col);
      if (pixel  == 255)
      {
        blackCounter = 0;
        candidate = Point(col, row);

      }
      if(pixel == 0)
      {
        blackCounter++;
      }

      if(blackCounter >  maxCounter)
      {
        fieldPoints.push_back(candidate);
        break;
      }
    }
  }

  return fieldPoints;
}

void loadLUT(string path)
{
  cv::FileStorage storage(path, cv::FileStorage::READ);
  storage["lut"] >> lut;
  storage.release();

}

void updateLUT(Mat &inImage, Mat &outImage)
{
  cv::Mat imageLut;
  cv::Vec3b colors;


  for(int i = 0; i < inImage.rows; i++)
    for(int j = 0; j < inImage.cols; j++)
    {
      cv::Vec3b val = inImage.at<cv::Vec3b>(i,j);

      int h = val.val[0] / 4;
      int s = val.val[1] / 4;
      int v = val.val[2] / 4;

      uchar space = lut.at<uchar>(0,h+s*bins+v*bins*bins);


      if(space == OTHER)
      {
        //        colors.val[0]=255;
        //        colors.val[1]=0;
        //        colors.val[2]=0;
        outImage.at<uchar>(i,j) = 0;
      }
      else if(space == GREEN)
      {
        //        colors.val[0]=0;
        //        colors.val[1]=255;
        //        colors.val[2]=0;
        outImage.at<uchar>(i,j) = 255;
      }
      else if(space == WHITE)
      {
        //        colors.val[0]=255;
        //        colors.val[1]=255;
        //        colors.val[2]=255;
        outImage.at<uchar>(i,j) = 255;
      }
      else if(space == BLACK)
      {
        //        colors.val[0]=0;
        //        colors.val[1]=0;
        //        colors.val[2]=0;
        outImage.at<uchar>(i,j) = 0;
      }

    }
}

void scanLine(Mat &input, vector<Point> &linePoint, vector<vector<Point> > &row_candidate, vector<vector<Point> > &col_candidate)
{
  int STEP = 40;

  for(int row = 0; row < input.rows; row++)
  {
    vector<Point> rowPoint;
    for(int col = 0; col < input.cols; col++)
    {
      int pixel = input.at<uchar>(row,col);

      if(pixel > 0)
      {
        rowPoint.push_back(Point(col, row));
      }
    }
    row_candidate.push_back(rowPoint);
  }

  for(int s_col = 0; s_col < input.cols; s_col+=STEP)
  {
    vector<Point> colPoint;
    bool start = false;
    int start_col = 0;

    for(int s_row = 0; s_row < input.rows; s_row++)
    {
      int pixel = input.at<uchar>(s_row, s_col);
      if((pixel == 255 ) && (start == false))
      {
        start_col = s_row;
        start = true;
      }

      if(pixel == 0 && start == true)
      {
        int end_col = s_row - 1;
        int middle = (start_col + end_col)/2;
        colPoint.push_back(Point(s_col, middle));
        start = false;

      }
    }
    col_candidate.push_back(colPoint);
  }

  for(int i = 0; i < row_candidate.size(); i++)
    for(int j = 0; j < col_candidate.size(); j++)
    {
      vector<Point> col = col_candidate[j];

      for(int x = 0; x < col.size(); x++)
        if(std::find(row_candidate[i].begin(), row_candidate[i].end(), col[x]) != row_candidate[i].end())
        {
          linePoint.push_back(col[x]);
        }

    }
}

void saveConfig(const string &path)
{
  YAML::Emitter out;

  Eigen::Vector3d fieldLow = Eigen::Vector3d(fieldRange.h0, fieldRange.s0,fieldRange.v0);
  Vec6 arp;
  arp.x = lineRange.h0; arp.y = lineRange.h1; arp.z = lineRange.s0;
  arp.r = lineRange.s1; arp.p = lineRange.v0; arp.w = lineRange.v1;
  Vec6 apr;
  apr.x = fieldYUV.h0; apr.y = fieldYUV.h1; apr.z = fieldYUV.s0;
  apr.r = fieldYUV.s1; apr.p = fieldYUV.v0; apr.w = fieldYUV.v1;

  Vec6 huhu;
  huhu.x = goalieHSV.h0; huhu.y = goalieHSV.h1; huhu.z = goalieHSV.s0;
  huhu.r = goalieHSV.s1; huhu.p = goalieHSV.v0; huhu.w = goalieHSV.v1;

  Vec6 haha;
  haha.x = goalieYUV.h0; haha.y = goalieYUV.h1; haha.z = goalieYUV.s0;
  haha.r = goalieYUV.s1; haha.p = goalieYUV.v0; haha.w = goalieYUV.v1;

  out << YAML::BeginMap;

  out << YAML::Key << "FieldLow" << YAML::Value << fieldLow;
  out << YAML::Key << "lineVal" << YAML::Value << arp;
  out << YAML::Key << "fieldYUV" << YAML::Value << apr;
  out << YAML::Key << "goalieHSV" << YAML::Value << huhu;
  out << YAML::Key << "goalieYUV" << YAML::Value << haha;

  out << YAML::EndMap;

  // output to file
  std::ofstream fout(path.c_str());
  fout << out.c_str();
}

void loadConfig(const string &path)
{
  YAML::Node conf = YAML::LoadFile(path.c_str());

  Eigen::Vector3d fieldLow = conf["FieldLow"].as<Eigen::Vector3d>();
  Vec6 linedata = conf["lineVal"].as<Vec6>();
  Vec6 fieldyuv = conf["fieldYUV"].as<Vec6>();
  Vec6 goaliehsv = conf["goalieHSV"].as<Vec6>();
  Vec6 goalieyuv = conf["goalieYUV"].as<Vec6>();

  fieldRange.low(fieldLow);
  lineRange.load(linedata);
  fieldYUV.load(fieldyuv);
  goalieHSV.load(goaliehsv);
  goalieYUV.load(goalieyuv);

}

void loadCameraCalibration(const std::string path, Mat &camInt, Mat &distCoeff, Mat &camProj)
{
  std::cout << "load camera data" << std::endl;
  YAML::Node conf = YAML::LoadFile(path.c_str());

  YAML::Node camera_mat = conf["camera_matrix"];
  int rows = camera_mat["rows"].as<int>();
  int cols = camera_mat["cols"].as<int>();

  vector<double> data;
  data.resize(rows*cols);

  if(data.size() == camera_mat["data"].size())
  {
    for (std::size_t i=0;i<camera_mat["data"].size();i++)
      data.at(i) = (camera_mat["data"][i].as<double>());

    camInt=Mat(data).reshape(0,rows);
  }
  data.clear();

  YAML::Node dist_coeff = conf["distortion_coefficients"];
  rows = dist_coeff["rows"].as<int>();
  cols = dist_coeff["cols"].as<int>();
  data.resize(rows*cols);

  if(data.size() == dist_coeff["data"].size())
  {
    for (std::size_t i=0;i<dist_coeff["data"].size();i++)
      data.at(i) = (dist_coeff["data"][i].as<double>());

    distCoeff=Mat(data).reshape(0,rows);
  }

  data.clear();

  YAML::Node proj_mat = conf["projection_matrix"];
  rows = proj_mat["rows"].as<int>();
  cols = proj_mat["cols"].as<int>();
  data.resize(rows*cols);

  if(data.size() == proj_mat["data"].size())
  {
    for (std::size_t i=0;i<proj_mat["data"].size();i++)
      data.at(i) = (proj_mat["data"][i].as<double>());

    camProj=Mat(data).reshape(0,rows);
  }

  std::cout << "Camera Matrix " << std::endl;
  std::cout << camInt << std::endl;
  std::cout << "Distortion Matrix" << std::endl;
  std::cout << distCoeff << std::endl;
  std::cout << "Projection Matrix" << std::endl;
  std::cout << camProj << std::endl;

}

void scanGoalie(std::vector<LineSegment> &lines, std::vector<cv::Point> &field_points, Mat &white_binary ,std::vector<LineSegment> &goal_posts)
{
  // find vertical lines
  // remove from sets

}

int main()
{
  const std::string path = "config.yaml";

  //    Mat image = imread("lapangan3.jpg", IMREAD_COLOR);
  VideoCapture cap("gazebo.mp4",cv::CAP_FFMPEG );
  if(!cap.isOpened())
  {
    cout << "ERROE" << endl;
    return -1;
  }

  cv::Mat camera_matrix;
  Mat distortionCoefficients;
  Mat camera_projection;

  //    loadLUT("lut.xml");
  loadCameraCalibration("head_camera.yaml", camera_matrix, distortionCoefficients, camera_projection);



  loadConfig(path);
  fieldTrackbar();
  //    lineTrackbar();
  goalieTrackbar();

  LineDetector *ld_ = new LineDetector;

  int counter = 0;

  while(1){
    cap >> image;


    RawBGRImg = image;

    cvtColor(RawBGRImg, RawHSVImg, CV_BGR2HSV);
    cvtColor(RawBGRImg, RawYUVImg, CV_BGR2YUV);

    /*
     * Find Contour, get field mask
     */

    Mat fieldBinary = Mat::zeros(RawHSVImg.size(), CV_8UC1);
    Mat fieldConvectHull = Mat::zeros(RawHSVImg.size(), CV_8UC1);

    getGreenField(RawBGRImg, fieldBinary);
    threshold(fieldBinary, fieldBinary, fieldRange.h0, 255, CV_THRESH_BINARY);
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    //                imshow("green out", fieldBinary);


    Mat yuv_out;
    inRange(RawYUVImg, Scalar(fieldYUV.h0, fieldYUV.s0, fieldYUV.v0), Scalar(fieldYUV.h1, fieldYUV.s1, fieldYUV.v1), yuv_out);

    bitwise_and(fieldBinary, yuv_out, fieldBinary);



    vector<vector<Point > > contours;
    vector<Point> ConvexHullPoints;
    findContours(fieldBinary,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    if(contours.size() > 0)
    {
      ConvexHullPoints =  contoursConvexHull(contours);
      if(ConvexHullPoints.size() > 0)
      {
        vector<vector<Point> > hullPoints = vector<vector<Point> > (1, ConvexHullPoints);

        //    polylines( image, ConvexHullPoints, true, Scalar(255,0,0), 2 );
        //    fillPoly(drawing, ConvexHullPoints, Scalar(255,255,255), 2);
        drawContours(fieldConvectHull, hullPoints, -1, Scalar(255), CV_FILLED);
      }
    }

    /*
     * Field Edge Point
     */

    Mat cleanField;
    cv::bitwise_and(fieldBinary, fieldConvectHull, cleanField);
    vector<Point> fieldBoundary = locateFieldPoints(cleanField);
    //    Vec4d fieldLines; Point pt1, pt2;

    //    for(int i = 0; i < fieldBoundary.size(); i++)
    //    {
    //        circle(image, fieldBoundary[i], 3, Scalar(100,255,100));
    //    }


    Mat out, lines_img;
    Mat lines_out, outer, outer_mask;
    Mat inv_field;
    Mat set = image.clone();
    bitwise_not(cleanField, inv_field);
    inRange(RawHSVImg, Scalar(lineRange.h0, lineRange.s0, lineRange.v0), Scalar(lineRange.h1, lineRange.s1, lineRange.v1), out);

    bitwise_and(inv_field, out, lines_img);
    set.copyTo(lines_out, lines_img);

    //    imshow("hehe", lines_out);

    bitwise_not(fieldConvectHull, outer_mask);
    set.copyTo(outer, outer_mask);

    /*
         * Goal Detector
         */
    Mat goalie_yuv, goalie_hsv, outer_hsv, outer_yuv;
    Mat goalie_mask, goalie_img;

    cvtColor(outer, outer_hsv, CV_BGR2HSV);
    cvtColor(outer, outer_yuv, CV_BGR2YUV);
    inRange(outer_hsv, Scalar(goalieHSV.h0, goalieHSV.s0, goalieHSV.v0), Scalar(goalieHSV.h1, goalieHSV.s1, goalieHSV.v1), goalie_hsv);
    inRange(outer_yuv, Scalar(goalieYUV.h0, goalieYUV.s0, goalieYUV.v0), Scalar(goalieYUV.h1, goalieYUV.s1, goalieYUV.v1), goalie_yuv);

    bitwise_and(goalie_hsv, goalie_yuv, goalie_mask);
    set.copyTo(goalie_img, goalie_mask);

    cvtColor(goalie_img, goalie_img, CV_BGR2GRAY);
    blur( goalie_img, goalie_img, Size(3,3) );
    Canny( goalie_img, goalie_img, 100, 300, 3 );

    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(goalie_img, linesP, 1, CV_PI/45, 60, 50, 40); // runs the actual detection
    //     Draw the lines
    vector<LineSegment> raw_goalie;
    for( size_t i = 0; i < linesP.size(); i++ )
    {
      Vec4i l = linesP[i];
      raw_goalie.push_back(LineSegment(l[0], l[1], l[2], l[3]));
      //            line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,123,255), 1, LINE_AA);
    }

    vector<LineSegment> filteredGoal;
    if(raw_goalie.size() > 1)
    {
      scanGoalie(raw_goalie, fieldBoundary, inv_field, filteredGoal);
    }

    /*
     * Field line
     */

    cvtColor(lines_out, lines_out, CV_BGR2GRAY);
    blur( lines_out, lines_out, Size(3,3) );

    /// Canny detector
    Canny( lines_out, lines_out, 100, 300, 3 );

    Mat cdstP = image.clone();

    // Probabilistic Line Transform
    //        vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(lines_out, linesP, 1, CV_PI/45, 60, 50, 40); // runs the actual detection
    // Draw the lines
    vector<LineSegment> rawLines;
    for( size_t i = 0; i < linesP.size(); i++ )
    {
      Vec4i l = linesP[i];
      rawLines.push_back(LineSegment(l[0], l[1], l[2], l[3]));
      //            line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,123,255), 1, LINE_AA);
    }

    vector<LineSegment> filteredLines;
    if(rawLines.size() > 1)
    {
      filteredLines = ld_->scanLine(rawLines);
    }

    if(filteredLines.size() > 0)
    {
      for( size_t i = 0; i < filteredLines.size(); i++ )
      {
        Point2d P1 = filteredLines[i].p1;
        Point2d P2 = filteredLines[i].p2;
        if(P1.x > cdstP.cols) P1.x = cdstP.cols;
        if(P1.x < 0) P1.x = 0;
        if(P2.x > cdstP.cols) P2.x = cdstP.cols;
        if(P2.x < 0) P2.x = 0;

        if(P1.y > cdstP.rows) P1.y = cdstP.rows;
        if(P1.y < 0) P1.y = 0;
        if(P2.y > cdstP.rows) P2.y = cdstP.rows;
        if(P2.y < 0) P2.y = 0;
        line( cdstP, filteredLines[i].p1, filteredLines[i].p2, Scalar(255,0,255), 2, LINE_AA);
      }
    }

    imshow("line hsv", cdstP);

    char key = waitKey(30);
    if(key == 27) break;
    else if(key == 's') saveConfig(path);

    //    waitKey(1);

    //    counter++;
  }

  std::cout << "finished"  <<std::endl;

  return 0;
}
