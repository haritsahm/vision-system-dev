#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <eigen3/Eigen/Eigen>
#include <yaml-cpp/yaml.h>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <random>
#include <math.h>

#define INF 1000000007
#define DEGREE2RADIAN M_PI/180
#define SQR(x) x*x

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

Mat image, RawHSVImg, RawBGRImg, undistImage;


class LineSegment
{
public:
  LineSegment(double a, double b,double c,double d)
  {p1.x=a;p1.y=b;p2.x=c;p2.y=d;}
  LineSegment()
  {p1 = p2 = Point2d(0,0);}
  LineSegment(const LineSegment &l)
  {
    p1 = l.p1; p2 = l.p2;
  }
  LineSegment(const Point2d a, const Point2d b)
  {
    p1 = a; p2 = b;
  }

  Point2d p1, p2;
  friend std::ostream& operator <<(std::ostream& os, LineSegment& l)
  {
    os << "Point A: " << l.p1 << ") | Point B: (" << l.p2 << ")";
    return os;
  }

  Point2d middleLineP()
  {
    return Point2d((p1.x + p2.x) / 2., (p1.y + p2.y) / 2.);
  }

  double lineLength()
  {
    double num1 = p2.x - p1.x;
    double num2 = p2.y - p1.y;
    return sqrt((double) num1 * (double) num1 + (double) num2 * (double) num2);
  }

  double lineSlope()
  {
    if (abs(p2.x - p1.x) < 0.00001)
    {
      return 0.00001;
    }
    return (p2.y - p1.y) / ((p2.x - p1.x)+ 1e-06);
  }


  //  bool operator< (const LineSegment &l) const
  //  {
  //    std::string rightStr = 	msgObj.m_MsgContent + msgObj.m_sentBy + msgObj.m_recivedBy;
  //    std::string leftStr = 	this->m_MsgContent + this->m_sentBy + this->m_recivedBy;
  //    return (leftStr < rightStr);
  //  }
};

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

hsvRangeC fieldRange;
hsvRangeC lineRange;

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

bool compareLine(LineSegment l1, LineSegment l2)
{
  if((l1.p1 == l2.p1) && (l1.p2 == l2.p2))
    return true;
  return false;
}

double bound(double t)
{
  if(t>1)
    return 1;
  else if(t<0)
    return 0;
  else return t;

}

double minDistance(LineSegment l1, LineSegment l2)
{
  Point2d d1 = l1.p2-l1.p1;
  Point2d d2 = l2.p2-l2.p1;
  Point2d d12 = l2.p1-l1.p1;

  double D1 = SQR(d1.x)+SQR(d1.y);
  double D2 = SQR(d2.x)+SQR(d2.y);

  double R = (d1.x*d2.x)+(d1.y*d2.y);
  double S1 = (d1.x*d12.x)+(d1.y*d12.y);
  double S2 = (d2.x*d12.x)+(d2.y*d12.y);

  //  cout << "d1" << d1 <<std::endl;
  //  cout << "d2" << d2 <<std::endl;
  //  cout << "d12" << d12 <<std::endl;

  //  cout << "D1" << D1 <<std::endl;
  //  cout << "D2" << D2 <<std::endl;
  //  cout << "R" << R <<std::endl;
  //  cout << "S1" << S1 <<std::endl;
  //  cout << "S2" << S2 <<std::endl;




  double denom = (D1*D2)-SQR(R);

  double t_,t,u;

  if(D1 == 0 || D2 == 0)
  {
    if(D1 != 0)
    {
      u=0;
      t=S1/D1;
      t=bound(t);

    }

    else if(D2 !=0)
    {
      t=0;
      u=-S2/D2;
      u=bound(u);
    }
    else
      t=u=0;

  }
  else if(denom==0)
  {
    t=0;
    u=-S2/D2;
    u=bound(u);
  }

  else
  {
    t=(S1*D2-S2*R)/denom;
    t=bound(t);
    u = (t*R-S2)/D2;
    u = bound(u);
  }

  Point2d nn = d1*t-d2*u-d12;
  double point_sq = SQR(nn.x)+SQR(nn.y);
  double dist = sqrt(point_sq);

  std::cout << "minDistance" << dist << std::endl;

  return dist;
}

double angleDiff(LineSegment &l1, LineSegment &l2)
{
  double m1 = l1.lineSlope();
  double m2 = l2.lineSlope();

  if(m1 == m2) return 1e-06;
  else if(m1*m2 == -1) return M_PI/2;


  double tan_th = (m2-m1)/(1+m1*m2+1e-06);

  double angle = atan(tan_th) * 180/M_PI;

  std::cout << "anglediff" << angle << std::endl;


  return abs(atan(tan_th));
}

LineSegment mergeLine(LineSegment &l1, LineSegment &l2)
{
//  std::cout << "merging line" << std::endl;

  LineSegment new_line;

  vector<LineSegment> ls; ls.push_back(l1);ls.push_back(l2);

  double slope = 0;

  Point2d Xm = l1.middleLineP();
  Point2d Ym = l2.middleLineP();

  double l1_length = l1.lineLength();
  double l2_length = l2.lineLength();
  double r = l1_length / ((l1_length + l2_length) + 1e-06);

//  std::cout << "calc r" << std::endl;


  Point2d P = r*Xm + (1-r)*Ym;

  if(l1_length >= l2_length)
    slope = l1.lineSlope();
  else
    slope = l2.lineSlope();

//  std::cout << "get slope" << std::endl;


  //line point eq     bs
  // y = mx + (P.y - slope*P.x)
  double bz = P.y - slope*P.x;

  double ort_slope = -1/slope;

  vector<Point2d> orth_lines;

  for(auto line_seg : ls)
  {

    double bs_l_p1 = line_seg.p1.y - ort_slope*line_seg.p1.x;
    double bs_l_p2 = line_seg.p2.y - ort_slope*line_seg.p2.x;

    double x1 = (bz-bs_l_p1)/((slope-ort_slope) +1e-06);
    double y1 = ort_slope*x1+ bs_l_p1;

    double x2 = (bz-bs_l_p2)/((slope-ort_slope) +1e-06);
    double y2 = ort_slope*x2 + bs_l_p2;

    orth_lines.push_back(Point2d(x1,y1));
    orth_lines.push_back(Point2d(x2,y2));
  }

//  std::cout << "get orth Lines" << std::endl;


  double dist = 0;
  for(int i = 0; i < orth_lines.size(); i++)
    for(int j = 0; j < orth_lines.size(); j++)
    {
      if(i == j) continue;

      LineSegment s(orth_lines[i], orth_lines[j]);
      if(s.lineLength() > dist)
      {
        dist = s.lineLength();
        new_line = s;
      }

    }

//  std::cout << "find largest" << std::endl;


  return new_line;
}

void findAndErase(vector<LineSegment> &data, LineSegment line)
{
//  std::cout << "find in segment data" << std::endl;
  int idx = 0;
  for(auto ls: data)
  {
    if(compareLine(ls, line))
      data.erase(data.begin()+idx);
    idx++;
  }

//  for (vector<LineSegment>::iterator it = data.begin(); it != data.end(); ++it) {
//    {
//      int index = std::distance(data.begin(), it);
//      std::cout << "line index " <<index << std::endl;

//      if(compareLine(*it, line))
//        data.erase(data.begin()+index);
//    }
//  }

}

vector<LineSegment> scanLine(vector<LineSegment> &rawLines)
{
  std::mt19937 generator;
  vector<LineSegment> N, M;
  N = rawLines;

  int size_M;

  do
  {
    M = N;
    size_M = M.size();
    N.clear();

//    std::cout << M.size() <<std::endl;

//    std::cout << "first do" << std::endl;

    while(!M.empty())
    {
      int rand_ = generator() % (M.size());
//      std::cout << "M Size " <<M.size() <<std::endl;
//      std::cout << "random nm " << rand_ << std::endl;
      LineSegment X;
      X= M.at(rand_);

//      std::cout << "remove X" << std::endl;


      findAndErase(M, X);

//      std::cout << "find and erase X" << std::endl;


      int size_M_hat = M.size();

//      std::cout << "get m hat size" << size_M_hat << std::endl;


      for(auto &Y: M)
      {
        if((angleDiff(X,Y) < (15*DEGREE2RADIAN)) && (minDistance(X,Y) < 100))
        {
          N.push_back(mergeLine(X,Y));
//          std::cout << "find erase merged Lines" << std::endl;
          findAndErase(M, Y);
          break;
        }
      }

//      std::cout << "finish merge lines" << std::endl;
//      std::cout << size_M_hat << std::endl;


      if(M.size() == size_M_hat)
        N.push_back(X);
    }
  }
  while(N.size() == size_M);

  return N;

}

void saveConfig(const string &path)
{
  YAML::Emitter out;

  Eigen::Vector3d fieldLow = Eigen::Vector3d(fieldRange.h0, fieldRange.s0,fieldRange.v0);
  Vec6 arp;
  arp.x = lineRange.h0; arp.y = lineRange.h1; arp.z = lineRange.s0;
  arp.r = lineRange.s1; arp.p = lineRange.v0; arp.w = lineRange.v1;

  out << YAML::BeginMap;

  out << YAML::Key << "FieldLow" << YAML::Value << fieldLow;
  out << YAML::Key << "lineVal" << YAML::Value << arp;

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

  fieldRange.low(fieldLow);
  lineRange.load(linedata);

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

  cv::Mat camera_matrix = (Mat1d(3, 3) << 296.374353, 0, 240.5, 0, 296.374353, 180.5, 0, 0, 1);
  Mat distortionCoefficients = (Mat1d(1, 5) << 0, 0, 0, 0, 0);

  loadLUT("lut.xml");

  loadConfig(path);
  //    fieldTrackbar();
  //    lineTrackbar();

  double dist = minDistance(LineSegment(0,1,1,4), LineSegment(1,0,2,4));
  std::cout << dist << std::endl;



  while(1){
    cap >> image;


    RawBGRImg = image;

    cvtColor(RawBGRImg, RawHSVImg, CV_BGR2HSV);
    /*
     * White from LUT
     */

    //        Mat whiteLUT = Mat::zeros(RawHSVImg.size(), CV_8UC1);
    //        updateLUT(RawHSVImg, whiteLUT);


    /*
     * Find Contour, get field mask
     */

    Mat fieldBinary = Mat::zeros(RawHSVImg.size(), CV_8UC1);
    Mat fieldConvectHull = Mat::zeros(RawHSVImg.size(), CV_8UC1);

    getGreenField(RawBGRImg, fieldBinary);
    threshold(fieldBinary, fieldBinary, 97, 255, CV_THRESH_BINARY);
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    erode(fieldBinary, fieldBinary, element);

    erode(fieldBinary, fieldBinary, element);


    vector<vector<Point > > contours;
    vector<Point> ConvexHullPoints;
    findContours(fieldBinary,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    if(contours.size() > 0)
    {
      ConvexHullPoints =  contoursConvexHull(contours);
      if(ConvexHullPoints.size() > 0)
      {
        vector<vector<Point> > hullPoints = vector<vector<Point> > (1, ConvexHullPoints);

        //            polylines( image, ConvexHullPoints, true, Scalar(255,0,0), 2 );
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


    /*
     * Field Line Poitns
     */

    Mat inv_field = Mat::zeros(cleanField.size(), CV_8UC1);
    bitwise_not(cleanField, inv_field);
    multiply(inv_field,fieldConvectHull,inv_field);

    //    cv::bitwise_and()


    //    vector<Point> linePoints;
    //    vector<vector<Point> > row_candidate, col_candidate;

    //    scanLine(inv_field, linePoints, row_candidate, col_candidate);

    //    for(int i = 0; i < linePoints.size(); i++)
    //        circle(image, linePoints[i], 3, Scalar(100,100,255));


    /*
       * Field Line
       */
    Mat out, lines_img;
    Mat lines_out;
    Mat set = image.clone();
    inRange(RawHSVImg, Scalar(lineRange.h0, lineRange.s0, lineRange.v0), Scalar(lineRange.h1, lineRange.s1, lineRange.v1), out);
    bitwise_and(inv_field, out, lines_img);
    set.copyTo(lines_out, lines_img);

    cvtColor(lines_out, lines_out, CV_BGR2GRAY);
    blur( lines_out, lines_out, Size(3,3) );

    /// Canny detector
    Canny( lines_out, lines_out, 100, 300, 3 );
    Mat canny_out =lines_out.clone();

    Mat cdstP = image.clone();

    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(lines_out, linesP, 1, CV_PI/180, 50, 50, 10 ); // runs the actual detection
    // Draw the lines
    vector<LineSegment> rawLines;
    for( size_t i = 0; i < linesP.size(); i++ )
    {
      Vec4i l = linesP[i];
      rawLines.push_back(LineSegment(l[0], l[1], l[2], l[3]));
//      line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,123,255), 1, LINE_AA);
    }

    vector<LineSegment> filteredLines;
    if(rawLines.size() > 1)
    {
      filteredLines = scanLine(rawLines);
    }

    if(filteredLines.size() > 0)
    {
      for( size_t i = 0; i < filteredLines.size(); i++ )
      {
        line( cdstP, filteredLines[i].p1, filteredLines[i].p2, Scalar(255,0,255), 2, LINE_AA);
      }
    }

    imshow("line hsv", cdstP);
    imshow("canny", canny_out);

    if(waitKey(30) == 27) break;
    else if(waitKey(30) == 's') saveConfig(path);

    //        waitKey();
  }

  return 0;
}
