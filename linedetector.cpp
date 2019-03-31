#include "linedetector.h"

using namespace cv;
using namespace std;

LineDetector::LineDetector()
{

}

vector<LineSegment> LineDetector::scanLine(vector<LineSegment> &rawLines)
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

        while(!M.empty())
        {
            int rand_ = generator() % (M.size());
            LineSegment X;
            X= M.at(rand_);
            findAndErase(M, X);
            int size_M_hat = M.size();

            for(auto &Y: M)
            {
                if((angleDiff(X,Y) < (15*DEGREE2RADIAN)) && (minDistance(X,Y) < 25))
                {
                    N.push_back(mergeLine(X,Y));
                    findAndErase(M, Y);
                    break;
                }
                else continue;
            }

            if(M.size() == size_M_hat)
                N.push_back(X);
        }
    }
    while(N.size() != size_M);

    return N;

}


void LineDetector::scanCircular(vector<LineSegment> &rawLines, bool &flagFound, Point2d &center)
{
    vector<Point2d> P;
    for(int i = 0; i < rawLines.size(); i++)
    {
        LineSegment X = rawLines[i];
        for(int j = 0; j < rawLines.size(); j++)
        {
            if(i==j) continue;
            LineSegment Y = rawLines[j];

            Point2d res;
            if(X.Intersect(Y, res))
                if(X.distancePoint(res) && Y.distancePoint(res))
                    P.push_back(res);
        }
    }

    if((P.size() > 5) && maxPtDistance(P) < 0.75)
    {
        flagFound = true;
        center = meanVector(P);
    }

    else
    {
        flagFound = false;
        center = Point2d(0,0);
    }
}

LineSegment LineDetector::mergeLine(LineSegment &l1, LineSegment &l2)
{
    LineSegment new_line;

    vector<LineSegment> ls; ls.push_back(l1);ls.push_back(l2);

    double slope = 0;

    Point2d Xm = l1.middleLineP();
    Point2d Ym = l2.middleLineP();

    double l1_length = l1.lineLength();
    double l2_length = l2.lineLength();
    double r = l1_length / ((l1_length + l2_length) + 1e-06);

    Point2d P = r*Xm + (1-r)*Ym;

    if(l1_length >= l2_length)
        slope = l1.lineSlope();
    else
        slope = l2.lineSlope();


    //line point eq
    // y-y1 = m(x-x1)
    double bz = P.y - slope*P.x;

    double ort_slope = -1/slope;

    vector<Point2d> orth_lines;

    for(auto line_seg : ls)
    {

        double bs_l_p1 = line_seg.p1.y - ort_slope*line_seg.p1.x;
        double bs_l_p2 = line_seg.p2.y - ort_slope*line_seg.p2.x;

        double x1 = -(bz-bs_l_p1)/((slope-ort_slope) +1e-06);
        double y1 = ort_slope*x1+ bs_l_p1;

        double x2 = -(bz-bs_l_p2)/((slope-ort_slope) +1e-06);
        double y2 = ort_slope*x2 + bs_l_p2;

        orth_lines.push_back(Point2d(x1,y1));
        orth_lines.push_back(Point2d(x2,y2));
    }

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

    return new_line;
}


double LineDetector::minDistance(LineSegment l1, LineSegment l2)
{
    Point2d d1 = l1.p2-l1.p1;
    Point2d d2 = l2.p2-l2.p1;
    Point2d d12 = l2.p1-l1.p1;

    double D1 = SQR(d1.x)+SQR(d1.y);
    double D2 = SQR(d2.x)+SQR(d2.y);

    double R = (d1.x*d2.x)+(d1.y*d2.y);
    double S1 = (d1.x*d12.x)+(d1.y*d12.y);
    double S2 = (d2.x*d12.x)+(d2.y*d12.y);

    double denom = (D1*D2)-SQR(R);

    double t,u;

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
        double uf=bound(u);
        if (uf != u)
        {
            t = (uf*R+S1)/D1;
            t = bound(t);
            u = uf;
        }

    }

    else
    {
        t=(S1*D2-S2*R)/denom;
        t=bound(t);
        u = (t*R-S2)/D2;
        double uf=bound(u);
        if (uf != u)
        {
            t = (uf*R+S1)/D1;
            t = bound(t);
            u = uf;
        }
    }

    Point2d nn = d1*t-d2*u-d12;
    double point_sq = nn.x*nn.x+nn.y*nn.y;
    double dist = sqrt(point_sq);
    return dist;
}

double LineDetector::bound(double t)
{
    if(t>1)
        return 1;
    else if(t<0)
        return 0;
    else return t;

}

bool LineDetector::compareLine(LineSegment l1, LineSegment l2)
{
    double a = l1.p1.x-l2.p1.x;
    double b = l1.p1.y-l2.p1.y;
    double c = l1.p2.x-l2.p2.x;
    double d = l1.p2.y-l2.p2.y;
    double val1 = sqrt(a*a+ b*b);
    double val2 = sqrt(c*c+ d*d);
    if(val1 < 40 && val2 < 40)
        return true;
    return false;
    //    if((l1.p1 == l2.p1) && (l1.p2 == l2.p2))
    //        return true;
    //    return false;
}

double LineDetector::angleDiff(LineSegment &l1, LineSegment &l2)
{
    double m1 = l1.lineSlope();
    double m2 = l2.lineSlope();
    double diff = abs(m1-m2);
    double mm = m1*m2;

    if(diff < 0.5 && diff > 0) return 1e-06;
    else if(mm <= -0.6 && mm >= -1) return M_PI/2;


    double tan_th = abs((m2-m1)/(1+(m1*m2)+1e-06));
    return atan(tan_th);
}

void LineDetector::findAndErase(vector<LineSegment> &data, LineSegment line)
{
    int idx = 0;
    for(auto ls: data)
    {
        if(compareLine(ls, line))
            data.erase(data.begin()+idx);
        idx++;
    }
}

double LineDetector::maxPtDistance(vector<Point2d> P)
{
    double max = 0;
    for(int i = 0; i < P.size(); i++)
    {
        Point2d X = P[i];
        for(int j = 0; j < P.size(); j++)
        {
            if(i == j) continue;
            Point2d Y = P[j];
            double num1 = Y.x - X.x;
            double num2 = Y.y - X.y;
            double dist =  sqrt((double) num1 * (double) num1 + (double) num2 * (double) num2);
            if(dist > max)
                max = dist;
        }
    }

    return max;
}

Point2d LineDetector::meanVector(vector<Point2d> P)
{
    double p_X = 0;
    double p_Y = 0;

    for(auto point : P)
    {
        p_X += point.x;
        p_Y += point.y;
    }

    p_X /= P.size();
    p_Y /= P.size();

    return Point2d(p_X, p_Y);
}

