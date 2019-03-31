#ifndef LINEDETECTOR_H
#define LINEDETECTOR_H

#define DEGREE2RADIAN M_PI/180
#define SQR(x) x*x

#include "linesegment.h"
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ximgproc.hpp>
#include <algorithm>
#include <random>
#include <math.h>
#include <vector>
#include <iostream>


class LineDetector
{
public:
    LineDetector();

    std::vector<LineSegment> scanLine(std::vector<LineSegment> &rawLines);
    void scanCircular(std::vector<LineSegment> &rawLines, bool &flagFound, cv::Point2d &center);
    bool compareLine(LineSegment l1, LineSegment l2);
    double bound(double t);
    double minDistance(LineSegment l1, LineSegment l2);
    double angleDiff(LineSegment &l1, LineSegment &l2);
    LineSegment mergeLine(LineSegment &l1, LineSegment &l2);
    void findAndErase(std::vector<LineSegment> &data, LineSegment line);
    double maxPtDistance(std::vector<cv::Point2d> P);
    cv::Point2d meanVector(std::vector<cv::Point2d> P);
};

#endif // LINEDETECTOR_H
