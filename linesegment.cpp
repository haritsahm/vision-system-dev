#include "linesegment.h"

using namespace cv;

LineSegment::LineSegment()
{
    p1 = p2 = Point2d(0,0);
}

LineSegment::LineSegment(double a, double b,double c,double d)
{
    p1.x=a;
    p1.y=b;
    p2.x=c;
    p2.y=d;
}

LineSegment::LineSegment(const LineSegment &l)
{
    p1 = l.p1;
    p2 = l.p2;
}
LineSegment::LineSegment(const Point2d a, const Point2d b)
{
    p1 = a; p2 = b;
}

cv::Point2d LineSegment::middleLineP()
{
    return cv::Point2d((p1.x + p2.x) / 2., (p1.y + p2.y) / 2.);
}

double LineSegment::lineLength()
{
    double num1 = p2.x - p1.x;
    double num2 = p2.y - p1.y;
    return sqrt((double) num1 * (double) num1 + (double) num2 * (double) num2);
}

double LineSegment::lineSlope()
{
    if (abs(p2.x - p1.x) < 0.00001)
    {
        return 0.00001;
    }
    return (p2.y - p1.y) / ((p2.x - p1.x)+ 1e-06);
}

void LineSegment::lineBisector(double &bias, double &slope)
{
    cv::Point2d midP = middleLineP();
    double slope_angle = lineSlope();
    double neg_reciprocal = -1/slope_angle;

    double b = midP.y - neg_reciprocal*midP.x;

    bias = b;
    slope = neg_reciprocal;
}

bool LineSegment::Intersect(LineSegment L, cv::Point2d &res)
{
    float s02_x, s02_y, s10_x, s10_y, s32_x, s32_y, s_numer, t_numer, denom, t;
    s10_x = p2.x - p1.x;
    s10_y = p2.y - p1.y;
    s32_x = L.p2.x - L.p1.x;
    s32_y = L.p2.y - L.p1.y;

    denom = s10_x * s32_y - s32_x * s10_y;
    if (denom == 0)
        return false; // Collinear
    bool denomPositive = denom > 0;

    s02_x = p1.x - L.p1.x;
    s02_y = p1.y - L.p1.y;
    s_numer = s10_x * s02_y - s10_y * s02_x;
    if ((s_numer < 0) == denomPositive)
        return false; // No collision

    t_numer = s32_x * s02_y - s32_y * s02_x;
    if ((t_numer < 0) == denomPositive)
        return false; // No collision

    if (((s_numer > denom) == denomPositive)
            || ((t_numer > denom) == denomPositive))
        return false; // No collision
    // Collision detected
    t = t_numer / denom;

    res.x = p1.x + (t * s10_x);

    res.y = p1.y + (t * s10_y);

    return true;
}

bool LineSegment::distancePoint(Point2d p)
{
    double num = abs((p2.y-p1.y)*p.x-(p2.x-p1.x)*p.y+p2.x*p1.y-p2.y*p1.x);
    double denom =  sqrt((p2.y-p1.y)*(p2.y-p1.y)+(p2.x-p1.x)*(p2.x-p1.x))+1e-06;

    double dist = num/denom;
    //assume dist already projected
    if(abs(dist - 0.75) < 0.04)
        return true;
    return false;
}
