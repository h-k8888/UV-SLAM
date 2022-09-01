#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <random>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv/cv.h>

#include "math.h"
#include "utility.h"
#include "highgui.h"
#include "ELSED.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;
using namespace cv;
using namespace line_descriptor;

typedef line_descriptor::BinaryDescriptor LineBD;
typedef line_descriptor::KeyLine LineKL;

struct Line
{
    Point2f start_xy;//endpoints xy in image frame, undistort
    Point2f end_xy;

    Point2f start_xy_visual, end_xy_visual;

    Point2f StartPt;
    Point2f EndPt;
    float lineWidth;
    Point2f Vp;

    Point2f Center;
    Point2f unitDir; // [cos(theta), sin(theta)]
    float length;
    float theta; //radius

    // para_a * x + para_b * y + c = 0
    float para_a;
    float para_b;
    float para_c;

    float image_dx;
    float image_dy;
    float line_grad_avg;

    float xMin;
    float xMax;
    float yMin;
    float yMax;
    unsigned short id;
    int colorIdx;

    bool start_predict_fail = false;
    bool end_predict_fail = false;

    bool updated_forwframe = false;
    bool extended = false;

    int num_untracked = 0;

    enum valid_type {
        valid = 1,
        too_short = 2,
        bad_ZNCC = 3,
        line_out_of_view = 4,
        endpoint_out_of_image = 5,
        bad_gradient_direction = 6,
        bad_prediction = 7,
        large_untracked = 8
    };

    valid_type is_valid = valid;

    deque<Point2f> sample_points_undistort; //纠正后图像中的采样点

};

class LineFeatureTracker
{
  public:
    LineFeatureTracker();

    void readImage4Line(const Mat &_img, double _cur_time);
    void imageUndistortion(Mat &_img, Mat &_out_undistort_img);
    void readIntrinsicParameter(const string &calib_file);
    void lineExtraction( Mat &cur_img, vector<LineKL> &_keyLine, Mat &_descriptor );
    void lineMergingTwoPhase( Mat &prev_img, Mat &cur_img, vector<LineKL> &prev_keyLine, vector<LineKL> &cur_keyLine, Mat &prev_descriptor, Mat &cur_descriptor, vector<DMatch> &good_match_vector );
    void lineMatching( vector<LineKL> &_prev_keyLine, vector<LineKL> &_curr_keyLine, Mat &_prev_descriptor, Mat &_curr_descriptor, vector<DMatch> &_good_match_vector);
    bool updateID(unsigned int i);
    void normalizePoints();

    void getVPHypVia2Lines(vector<KeyLine> cur_keyLine, vector<Vector3d> &para_vector, vector<double> &length_vector, vector<double> &orientation_vector,
                           std::vector<std::vector<Vector3d> > &vpHypo );
    void getSphereGrids(vector<KeyLine> cur_keyLine, vector<Vector3d> &para_vector, vector<double> &length_vector, vector<double> &orientation_vector,
                        std::vector<std::vector<double> > &sphereGrid );
    void getBestVpsHyp( std::vector<std::vector<double> > &sphereGrid, std::vector<std::vector<Vector3d> >  &vpHypo, std::vector<Vector3d> &vps  );
    void lines2Vps(vector<KeyLine> cur_keyLine, double thAngle, std::vector<Vector3d> &vps, std::vector<std::vector<int> > &clusters, vector<int> &vp_idx);
    void drawClusters( cv::Mat &img, std::vector<KeyLine> &lines, std::vector<std::vector<int> > &clusters );

    void lineRawResolution( Mat &cur_img, vector<LineKL> &predict_keyLines);

    camodocal::CameraPtr m_camera;
    camodocal::PinholeCameraPtr pinhole_camera;

    Mat Camera_Matrix = Mat(3,3,CV_32FC1, Scalar::all(0.0));
    Mat Discotrion_Coefficients = Mat(1, 4, CV_32FC1);
    Mat new_camera_matrix, undist_map1, undist_map2;

    Mat prev_img, curr_img, forw_img;
    Mat curr_descriptor, m_matched_descriptor;

    //////hk
    cv::Mat dxImg, dyImg, gradient_dir; // degree
    int allfeature_cnt;                  // 用来统计整个地图中有了多少条线，它将用来赋值
    int last_feature_count = -1;
    vector<Line> curr_line, forw_line;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;//线特征端点坐标
    vector<Line> lines_predict;
    vector< int > lineID; //当前帧索引 --> 全局索引
    vector< int > prev_lineID; //当前帧索引 --> 全局索引
    //NMS extend length in ELSED(vertical)
    int nms_extend = 15;
    //max lines in every frame
    int max_lines_num = 50;

    ////////hk

    vector<Point2f> curr_start_pts, curr_end_pts;
    vector<line_descriptor::KeyLine> curr_keyLine, forw_keyLine, m_matched_keyLines;

    vector<int> ids, tmp_ids; // set every value of tmp_ids to -1 when the new lines are extracted
    vector<int> track_cnt, tmp_track_cnt; // not sure // initial value: 1
    vector<int> vp_ids, tmp_vp_ids;
    vector<Vector3d> vps;
    vector<Point2f> start_pts_velocity, end_pts_velocity;
    vector<Point2f> prev_start_un_pts, curr_start_un_pts, prev_end_un_pts, curr_end_un_pts; // not sure
    Vector3d vp;


    void getUndistortEndpointsXY(const vector<Line>& lines, vector<cv::Point2f>& endpoints);
    void getUndistortSample(const vector<Line>& lines, vector<cv::Point2f>& points, std::vector<int>& point2lineIdx);
    void initLinesPredict();
    void rejectWithF(vector<uchar>& status);
    void markFailEndpoints(const vector<uchar>& status);
    void markFailSamplePoints(const vector<uchar>& status, vector<int>& point2lineIdx);
    void checkEndpointsAndUpdateLinesInfo();
    void checkOpticalFlowEndpoints(vector<Line>& lines, const vector<Point2f>& endpoints);
    void checkAndUpdateEndpoints(vector<Line>& lines);

    void MakeALine( cv::Point2f start_pts, cv::Point2f end_pts, const int& rows, const int& cols, Line& line);
    void MakeALine(cv::Point2f start_pts, cv::Point2f end_pts, Line& line);
    void reduceLine(vector<Line> &lines, vector<int>& IDs);
    void fitLinesBySamples(std::vector<Line>& lines);
    void extractELSEDLine(const Mat &img, vector<Line>& lines, vector<Line>& lines_exist);

    bool checkMidPointDistance(const cv::Point2f& start_i, const cv::Point2f& end_i,
                               const cv::Point2f& start_j,const cv::Point2f& end_j, const float threshold = 5.0);
    float point2line(const cv::Point2f& p, const cv::Point2f& start_l, const cv::Point2f& end_l) {
        const Point2f sp(p - start_l);
        const Point2f se(end_l - start_l);
        return abs(sp.cross(se) / norm(se));
    }
    void updateTrackingLinesAndID();
    void checkGradient(std::vector<Line>& lines, const int& start_idx = 0);
    bool largeGradientAngle(const float& line_angle, const float& x, const float& y, const float& threshold = 20.0);

    void Line2KeyLine(const vector<Line>& lines_in, vector<LineKL>& lines_out);
    void KeyLine2Line(const vector<LineKL>&lines_in, vector<Line>&  lines_out);

    void correctNegativeXY(cv::Point2f& start_pts, cv::Point2f& end_pts);
    void correctOutsideXY(cv::Point2f& start_pts, cv::Point2f& end_pts, const int& rows, const int& cols);

//    void LineFeatureTracker::visualize_line(const Mat &imageMat1, const FrameLinesPtr frame, const string &name, const bool show_NMS_area);
    void visualize_line(const Mat &imageMat1, const std::vector<Line>& lines, const string &name, const bool show_NMS_area = false);
    void DrawRotatedRectangle(cv::Mat& image, const cv::Point2f& centerPoint, const cv::Size& rectangleSize,
                              const float& rotationDegrees, const int& val = 144);
    void fillRectangle(Mat& img, const Point2f* pts, int npts, const void* color);


    static int n_id;
    static int vp_id;
    int image_id = 0;

    Utility util;


    /// FOR KALMAN
    // States are position and velocity in X and Y directions; four states [X;Y;dX/dt;dY/dt]
    CvPoint pt_Prediction, pt_Correction;

    // Measurements are current position of the mouse [X;Y]
    CvMat* measurement = cvCreateMat(2, 1, CV_32FC1);

    // dynamic params (4), measurement params (2), control params (0)
//    CvKalman* kalman  = cvCreateKalman(4, 2, 0);

    void CannyDetection(Mat &src, vector<line_descriptor::KeyLine> &keylines);
    bool getPointChain( const Mat & img, const Point2f pt, Point2f * chained_pt, int & direction, int step );
    void extractSegments( vector<Point2f> * points, vector<line_descriptor::KeyLine> * keylines);
    double distPointLine( const Mat & p, Mat & l );
    void additionalOperationsOnSegments(Mat & src, line_descriptor::KeyLine * kl);
    void pointInboardTest(Mat & src, Point2f * pt);
    bool mergeSegments(line_descriptor::KeyLine * kl1, line_descriptor::KeyLine * kl2, line_descriptor::KeyLine * kl_merged );
    void mergeLines(line_descriptor::KeyLine * kl1, line_descriptor::KeyLine * kl2, line_descriptor::KeyLine * kl_merged);
    void HoughDetection(Mat &src, vector<line_descriptor::KeyLine> &keylines);
    void OpticalFlowExtraction( Mat &prev_img, Mat &cur_img,
                               vector<LineKL> &prev_keyLine, vector<LineKL> &cur_keyLine,
                               Mat &prev_descriptor, Mat &cur_descriptor);
    double Union_dist(VectorXd a, VectorXd b);
    double Intersection_dist(VectorXd a, VectorXd b);
    VectorXd Union(VectorXd a, VectorXd b);
    VectorXd Intersection(VectorXd a, VectorXd b);
    double SafeAcos (double x);
    void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove) ;
    void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove);


    template<class tType>
    void incidentPoint( tType * pt, Mat & l );

    int threshold_length = 20;
    double threshold_dist = 1.5;
    int imagewidth, imageheight;
    int ROW_MARGIN = 15;
    int COL_MARGIN = 20;
};
