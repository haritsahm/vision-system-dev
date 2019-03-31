#ifndef LINESEGMENT_H
#define LINESEGMENT_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <random>
#include <math.h>
#include <vector>
#include <iostream>

class LineSegment
{
public:
    cv::Point2d p1, p2;

    LineSegment();
    LineSegment(const cv::Point2d a, const cv::Point2d b);
    LineSegment(double a, double b,double c,double d);
    LineSegment(const LineSegment &l);

    cv::Point2d middleLineP();
    double lineLength();
    double lineSlope();
    void lineBisector(double &bias, double &slope);
    bool Intersect(LineSegment L, cv::Point2d &res);
    bool distancePoint(cv::Point2d p);

    friend std::ostream& operator <<(std::ostream& os, LineSegment& l)
    {
        os << "Point A: " << l.p1 << ") | Point B: (" << l.p2 << ")";
        return os;
    }
};

#endif // LINESEGMENT_H
