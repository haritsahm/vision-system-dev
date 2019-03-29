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

Mat image, RawHSVImg, RawBGRImg, undistImage;


struct LineSegment
{
    double x1,y1;
    double x2,y2;
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

    static bool decode(const Node& node,Vec6& rhs) {
        if(!node.IsSequence() || node.size() != 6) {
            return false;
        }

        rhs.x = node[0].as<double>();
        rhs.y = node[1].as<double>();
        rhs.z = node[2].as<double>();
        rhs.r = node[3].as<double>();
        rhs.p = node[4].as<double>();
        rhs.w = node[5].as<double>();
        return true;
    }
};
}

YAML::Emitter& operator << (YAML::Emitter& out, const Eigen::Vector3d& v) {
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x() << v.y() << v.z() << YAML::EndSeq;
    return out;
}

YAML::Emitter& operator << (YAML::Emitter& out, const Vec6& v) {
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
        for( size_t i = 0; i < linesP.size(); i++ )
        {
            Vec4i l = linesP[i];
            line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
        }

        imshow("clean_white", cleanField);
        imshow("Image ", image);
        imshow("inv_field", inv_field);
        imshow("line hsv", cdstP);
        imshow("canny", canny_out);
        imshow("mask", cdstP);

        if(waitKey(30) == 27) break;
        else if(waitKey(30) == 's') saveConfig(path);

        //        waitKey();
    }

    return 0;
}
