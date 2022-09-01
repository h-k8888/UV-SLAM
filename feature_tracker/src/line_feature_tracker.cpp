#include "line_feature_tracker.h"

#include <chrono>

int LineFeatureTracker::n_id = 0;
int LineFeatureTracker::vp_id = 0;
unsigned int frame_count = 0;

///// TMP OUT FOR DEBUG
unsigned int matched_count = 0;
int keyLine_id = 0;
int img_num = 0;

upm::ELSED elsed;
const int LINE_MAX_UNTRACKED = 5;

bool inBorder(const cv::Point2f &pt, const int& row, const int& col)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

void visualize_line_samples_undistort(const Mat &imageMat1, const std::vector<Line>& lines_in, const string &name)
{
    //	Mat img_1;
    cv::Mat img1;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < lines_in.size(); ++k)
    {
        Line line1 = lines_in[k];  // trainIdx
        unsigned int r = lowest + int(rand() % range);
        unsigned int g = lowest + int(rand() % range);
        unsigned int b = lowest + int(rand() % range);

        //line
        cv::Point2f startPoint = line1.start_xy;
        cv::Point2f endPoint = line1.end_xy;
        cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b),2 ,8);

        //sample points
        for (const Point2f& p : line1.sample_points_undistort)
            cv::circle(img1, p, 2, cv::Scalar(r, g, b), 4);

        //start mid end point
        // b g r
//        cv::circle(img1, startPoint, 2, cv::Scalar(255, 0, 0), 5);
//        cv::circle(img1, endPoint, 2, cv::Scalar(0, 0, 255), 5);
//        cv::circle(img1, 0.5 * startPoint + 0.5 * endPoint, 2, cv::Scalar(0, 255, 0), 5);
    }

    imshow(name, img1);
    waitKey(1);
}


LineFeatureTracker::LineFeatureTracker()
{
    allfeature_cnt = 0;
}

void LineFeatureTracker::readImage4Line(const Mat &_img, double _cur_time)
{
    TicToc t_r;
    Mat img;
    if (EQUALIZE) //always equlize true
    {
        Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);

        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    /// raw image undistortion
    Mat undistort_img, undistort_img1;
    imageUndistortion(img, undistort_img); //undistort_img 垂直合并原图像和畸变纠正后的图像
    if (forw_img.empty())
        prev_img = curr_img = forw_img = undistort_img.clone();
    else
        forw_img = undistort_img.clone();

    if(DIST_K1 > 0)
    {
        forw_img = forw_img.rowRange(ROW_MARGIN, ROW - ROW_MARGIN);
        forw_img = forw_img.colRange(COL_MARGIN, COL - COL_MARGIN);
    }

//    Mat img1 = curr_img.clone();
//    Mat img2 = forw_img.clone();
//    Mat merged_img;
//    cvtColor(img1, img1, CV_GRAY2BGR);
//    cvtColor(img2, img2, CV_GRAY2BGR);
//    cvtColor(merged_img, merged_img, CV_GRAY2BGR);

    vector<Vector3d> para_vector;
    vector<double> length_vector;
    vector<double> orientation_vector;
    vector<std::vector<Vector3d> > vpHypo;
    vector<std::vector<double> > sphereGrid;
    vector<Vector3d> tmp_vps;
    vector<Vector3d> cluster_vps;
    vector<int> line_vp_ids;
    vector<vector<int>> clusters;
    vector<int> local_vp_ids;
    double thAngle = 1.0 / 180.0 * CV_PI;

    TicToc t_gr;
    cv::Sobel(forw_img, dxImg, CV_16SC1, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
    cv::Sobel(forw_img, dyImg, CV_16SC1, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
//    ROS_DEBUG("cv::Sobel cost: %f ms", t_gr.toc());

    //上一帧已探测到直线
    if(curr_keyLine.size()>0)//todo curr_line
    {
        last_feature_count = allfeature_cnt - 1; //上一帧最大特征id

        //get current frame endpoints
        int num_lines_curr = curr_line.size();
        getUndistortEndpointsXY(curr_line, cur_pts);
        ROS_DEBUG("cur_pts.size(): %d", cur_pts.size());
//        visualize_line_samples_undistort(curr_img, curr_line, "last frame");
//        visualize_line_samples_undistort(forw_img, curr_line, "curr_line forw frame");

        // add sample points to cur_pts
        std::vector<Point2f> cur_samples;
        std::vector<int> point2line_idx; //point  --->  line local index
        getUndistortSample(curr_line, cur_samples, point2line_idx);
        cur_pts.insert(cur_pts.end(), cur_samples.begin(), cur_samples.end());

        TicToc t_o; // optical flow predict line edpoints
        vector<uchar> status;
        vector<float> err;
        //predict endpoints using LK optical flow
        if (!cur_pts.empty())
        {
            forw_pts.clear();
            cv::calcOpticalFlowPyrLK(curr_img, forw_img, cur_pts, forw_pts, status, err,
                                     cv::Size(21, 21), 3);
        }
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());

        initLinesPredict(); // set lines_predict
        vector<uchar> status_f;
        rejectWithF(status_f);
        ROS_ASSERT(status.size() == status_f.size());
        for (int i = 0; i < (int)status.size(); ++i) {
            if (!status_f[i])
                status[i] = 0;
        }
        markFailEndpoints(status);
        ROS_DEBUG("markFailSamplePoints");
        markFailSamplePoints(status, point2line_idx);
        ROS_DEBUG("markFailSamplePoints, done");
        checkEndpointsAndUpdateLinesInfo();
//        visualize_line_samples_undistort(forw_img, lines_predict, "optical before reduce line");

        //Remove lines with insufficient sampling points
        reduceLine(lines_predict, prev_lineID);
//        visualize_line_samples_undistort(forw_img, lines_predict, "optical after reduce line");


        fitLinesBySamples(lines_predict);
//        visualize_line_samples_undistort(forw_img, lines_predict, "fit line curr --> forw");


        //    lineExtraction(img, lsd);
        ROS_DEBUG("ELSED: begin.");
        TicToc t_elsed;
        //enough num of lines
//    if (lines_predict.size() < max_lines_num)
//        visualize_line_samples_undistort(forw_img, lines_predict, "optical");
//        lines_predict.clear();
//        prev_lineID.clear();
//        tmp_track_cnt.clear();
        extractELSEDLine(forw_img, forw_line, lines_predict);
        ROS_DEBUG("line ELSED costs: %fms, new lines: %d.", t_elsed.toc(), forw_line.size());
//        visualize_line(forw_img, forw_line, "new lines forwframe");
//        visualize_line(forw_img, lines_predict, "lines predict", false);

        //reduce lines with larage num_untracked
        reduceLine(lines_predict, prev_lineID);

        ROS_DEBUG("updateTrackingLinesAndID");
        updateTrackingLinesAndID();
//        TicToc t_vis_line_new;
//        visualize_line(forwframe_->img, forwframe_, "lines forward");
//        visual_time += t_vis_line_new.toc();
        ROS_DEBUG("updateTrackingLinesAndID, done");

        TicToc t_gr;
        ROS_DEBUG("check Sample Gradient");
        checkGradient(forw_line);
//        reduceLine(lines_predict, prev_lineID);
        ROS_DEBUG("check Sample Gradient: %f ms", t_gr.toc());

        Line2KeyLine(forw_line, forw_keyLine);
        tmp_ids = lineID;

//        vector<uchar> status;
//        vector<DMatch> good_match_vector, good_match_vector2;
        vector<Point2f> forw_start_pts, forw_end_pts;
//        Mat forw_descriptor;
//
//        TicToc t_r;
//        //线特征提取，计算LBD描述子
//        lineExtraction(forw_img, forw_keyLine, forw_descriptor);
//        double t_extraction = t_r.toc();
//        //LBD描述子匹配线特征
//        lineMatching(curr_keyLine, forw_keyLine, curr_descriptor, forw_descriptor, good_match_vector);
//        double t_matching = t_r.toc() - t_extraction;
////        cout << t_extraction << ", " << t_matching << endl;
////        double t_linematching = t_r.toc() - t_lineextraction;
////        lineMergingTwoPhase( curr_img, forw_img, curr_keyLine, forw_keyLine, curr_descriptor, forw_descriptor, good_match_vector );
////        double t_linemerging = t_r.toc() - t_linematching;

        //如果新一帧有线特征
        if(forw_keyLine.size() > 1)
        {
            //计算曼哈顿世界的3个消失点
            getVPHypVia2Lines(forw_keyLine, para_vector, length_vector, orientation_vector, vpHypo);
//            ROS_DEBUG("getVPHypVia2Lines");
            getSphereGrids(forw_keyLine, para_vector, length_vector, orientation_vector, sphereGrid );
//            ROS_DEBUG("getSphereGrids");
            getBestVpsHyp(sphereGrid, vpHypo, tmp_vps);
//            ROS_DEBUG("getBestVpsHyp");
            lines2Vps(forw_keyLine, thAngle, tmp_vps, clusters, local_vp_ids);
//            ROS_DEBUG("lines2Vps");

        }

//        tmp_track_cnt.clear();
//        tmp_ids.clear();
        tmp_vp_ids.push_back(-1);
        start_pts_velocity.clear();
        end_pts_velocity.clear();
        vps.clear();

        if(forw_keyLine.size() > 0)        //如果新一帧有线特征
        {
            //遍历所有线特征，记录起点、终点、速度等信息
            for(int i=0; i<forw_keyLine.size(); i++)
            {
                cv::Point2f start_pts = forw_keyLine[i].getStartPoint();
                cv::Point2f end_pts = forw_keyLine[i].getEndPoint();
                forw_start_pts.push_back( start_pts );
                forw_end_pts.push_back(end_pts);

                if(!local_vp_ids.empty())
                {
                    if(local_vp_ids[i] == 3)
                        vps.push_back(Vector3d(0.0,0.0,0.0));
                    else
                        vps.push_back(tmp_vps[local_vp_ids[i]]/tmp_vps[local_vp_ids[i]](2)); //坐标归一化
                }

                //TODO initialize tmp_index
//                tmp_track_cnt.push_back(1);
//                tmp_ids.push_back(-1);
                tmp_vp_ids.push_back(-1);
                start_pts_velocity.push_back({0, 0});
                end_pts_velocity.push_back({0, 0});
            }
        }
//        vector<int> local2id;
//
//        //bring ids & cnt of trackted lines and update tmp_index & tmp_cnt
//        unsigned int num_tracked_line = 0;
//        if(good_match_vector.size() > 0)
//        {
//            for(int i=0; i< good_match_vector.size(); i++)
//            {
////                tmp_vp_ids.at(good_match_vector.at(i).trainIdx)=vp_ids.at(good_match_vector.at(i).queryIdx);
//                tmp_ids.at(good_match_vector.at(i).trainIdx)=ids.at(good_match_vector.at(i).queryIdx); //记录线 id
//                tmp_track_cnt.at(good_match_vector.at(i).trainIdx)=track_cnt.at(good_match_vector.at(i).queryIdx)+1; //追踪次数+1
//            }
//        }


//        ids.clear();
        track_cnt.clear();
        vp_ids.clear();

//        ids = tmp_ids;
//        ids = prev_lineID;
        track_cnt = tmp_track_cnt;
//        vp_ids = local_vp_ids;

//        cout << vp_ids.size() << ", " << forw_keyLine.size() << endl;

//        imshow("1", tmp_img);
//        waitKey(1);
        curr_img = forw_img;
        curr_start_pts = forw_start_pts;
        curr_end_pts = forw_end_pts;
        curr_keyLine = forw_keyLine;

        curr_line = forw_line;
//        curr_descriptor = forw_descriptor.clone();
    }
    else
    {
        curr_img = forw_img.clone();

        ROS_DEBUG("ELSED: begin.");
        TicToc t_elsed;
//        lineExtraction(curr_img, curr_keyLine, curr_descriptor);
//        KeyLine2Line(curr_keyLine, curr_line);
        lines_predict.clear();
        extractELSEDLine(curr_img, curr_line, lines_predict);
        if (curr_line.size() > max_lines_num)
            curr_line.resize(max_lines_num);
        Line2KeyLine(curr_line, curr_keyLine);
        ROS_DEBUG("line ELSED costs: %fms, lines: %d.", t_elsed.toc(), curr_line.size());

        TicToc t_gr;
        ROS_DEBUG("check Sample Gradient");
        checkGradient(curr_line);
//        reduceLine(lines_predict, prev_lineID);
        ROS_DEBUG("check Sample Gradient: %f ms", t_gr.toc());
//        visualize_line_samples_undistort(curr_img, curr_line, "first frame");

        if(curr_keyLine.size() > 1)
        {
            getVPHypVia2Lines(curr_keyLine, para_vector, length_vector, orientation_vector, vpHypo);
//            ROS_DEBUG("getVPHypVia2Lines, done");
            getSphereGrids(curr_keyLine, para_vector, length_vector, orientation_vector, sphereGrid );
//            ROS_DEBUG("getSphereGrids, done");
            getBestVpsHyp(sphereGrid, vpHypo, tmp_vps);
//            ROS_DEBUG("getBestVpsHyp, done");
            lines2Vps(curr_keyLine, thAngle, tmp_vps, clusters, local_vp_ids);
//            ROS_DEBUG("lines2Vps, done");
        }
//        drawClusters(img2, curr_keyLine, clusters);
        ROS_DEBUG("compute vanishing point, done.");


        curr_start_pts.clear();
        curr_end_pts.clear();
        tmp_track_cnt.clear();
        tmp_ids.clear();
        start_pts_velocity.clear();
        end_pts_velocity.clear();
        vps.clear();
        vp_ids.clear();

        for(int i=0; i< curr_keyLine.size(); i++)
        {
            cv::Point2f start_pts =  curr_keyLine[i].getStartPoint();
            cv::Point2f end_pts = curr_keyLine[i].getEndPoint();
            curr_start_pts.push_back(start_pts);
            curr_end_pts.push_back(end_pts);

            if(!local_vp_ids.empty())
            {
//                vps.push_back(Vector3d(0,0,0));
////                vp_ids.push_back(local_vp_ids[i]);
                if(local_vp_ids[i] == 3)
                    vps.push_back(Vector3d(0.0,0.0,0.0));
                else
                    vps.push_back(tmp_vps[local_vp_ids[i]]/tmp_vps[local_vp_ids[i]](2));
            }

            //TODO initialize tmp_index
            tmp_track_cnt.push_back(1);
            tmp_ids.push_back(-1);
            start_pts_velocity.push_back({0, 0});
            end_pts_velocity.push_back({0, 0});
        }

        Mat tmp_img = curr_img.clone();
        cvtColor(curr_img, tmp_img, CV_GRAY2BGR);

        ids.clear();
        track_cnt.clear();
        ids = tmp_ids;
        track_cnt = tmp_track_cnt;
        ROS_DEBUG("first frame, done");
    }

//    float count = 0;
//    for(int i = 0; i < track_cnt.size(); i++)
//    {
//        count += track_cnt[i];
//    }
    prev_start_un_pts = curr_start_un_pts;
    prev_end_un_pts = curr_end_un_pts;

    normalizePoints(); //转换到相机归一化坐标系
    int frame_index = 0;
//    cout << t_r.toc() << endl;
}

bool FindMatchedLine( LineKL query_line, LineKL train_line,
                      double min_diff_length, double min_diff_distance, double min_diff_angle ){

    // check the length of two lines
    double length_query = norm( Mat(query_line.getStartPoint()), Mat(query_line.getEndPoint()) );
    double length_train = norm( Mat(train_line.getStartPoint()), Mat(train_line.getEndPoint()) );

    if( fabs(length_query - length_train) > min_diff_length )
        return false;

    // check distance between two mid points each lines
    Point2f mid_p_query = (query_line.getStartPoint() + query_line.getEndPoint())/2.;
    Point2f mid_p_train = (train_line.getStartPoint() + train_line.getEndPoint())/2.;

    double dist = norm( Mat(mid_p_query), Mat(mid_p_train));

    if ( dist > min_diff_distance )
        return false;

    // check angle between two vectors
    Point2f v_query = query_line.getEndPoint() - query_line.getStartPoint();
    Point2f v_train = train_line.getEndPoint() - train_line.getStartPoint();

    double diff_angle = acos(v_query.dot(v_train) / (length_query*length_train));

    if ( diff_angle > min_diff_angle )
        return false;

    return true;
}

void LineFeatureTracker::lineMatching( vector<LineKL> &_prev_keyLine, vector<LineKL> &_curr_keyLine, Mat &_prev_descriptor,
                                      Mat &_curr_descriptor, vector<DMatch> &_good_match_vector)
{
    Ptr<BinaryDescriptorMatcher> bd_match = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    vector<vector<DMatch> > matches_vector;
    auto prev_descriptor = _prev_descriptor.clone();

    std::vector<DMatch> matches;
    bd_match->match(prev_descriptor, _curr_descriptor, matches);
    std::vector<DMatch> good_matches;
    for ( int i = 0; i < (int) matches.size(); i++ )
    {
        if(matches[i].queryIdx > _prev_keyLine.size() || matches[i].trainIdx > _curr_keyLine.size())
            continue;
        if(norm(Mat(_prev_keyLine[matches[i].queryIdx].getStartPoint()),
                Mat(_curr_keyLine[matches[i].trainIdx].getStartPoint())) > 30 ||
           norm(Mat(_prev_keyLine[matches[i].queryIdx].getEndPoint()),
                Mat(_curr_keyLine[matches[i].trainIdx].getEndPoint())) > 30)
            continue;

        good_matches.push_back( matches[i] );
    }
    _good_match_vector = good_matches;


//    cout << _good_match_vector.size() << ", " << good_matches.size() << endl;


//    cv::Mat outImg;
//    cv::Mat scaled1, scaled2;
//    std::vector<char> mask( matches.size(), 1 );
//    Mat pre_img, cur_img;
//    cvtColor(curr_img, pre_img, CV_GRAY2BGR);
//    cvtColor(forw_img, cur_img, CV_GRAY2BGR);
//    drawLineMatches( pre_img, lbd_octave1, cur_img, lbd_octave2, good_matches, outImg, Scalar::all( -1 ), Scalar::all( -1 ), mask,
//                       DrawLinesMatchesFlags::DEFAULT );
//    imshow( "Matches", outImg );
//    waitKey(1);
}

LineKL MakeKeyLine( cv::Point2f start_pts, cv::Point2f end_pts, size_t cols ){
    LineKL keyLine;
    //    keyLine.class_id = 0;
    //    keyLine.numOfPixels;

    // Set start point(and octave)
    if(start_pts.x > end_pts.x)
    {
        cv::Point2f tmp_pts;
        tmp_pts = start_pts;
        start_pts = end_pts;
        end_pts = tmp_pts;
    }

    keyLine.startPointX = (int)start_pts.x;
    keyLine.startPointY = (int)start_pts.y;
    keyLine.sPointInOctaveX = start_pts.x;
    keyLine.sPointInOctaveY = start_pts.y;

    // Set end point(and octave)
    keyLine.endPointX = (int)end_pts.x;
    keyLine.endPointY = (int)end_pts.y;
    keyLine.ePointInOctaveX = end_pts.x;
    keyLine.ePointInOctaveY = end_pts.y;

    // Set angle
    keyLine.angle = atan2((end_pts.y-start_pts.y),(end_pts.x-start_pts.x));

    // Set line length & response
    keyLine.lineLength = keyLine.numOfPixels = norm( Mat(end_pts), Mat(start_pts));
    keyLine.response = norm( Mat(end_pts), Mat(start_pts))/cols;

    // Set octave
    keyLine.octave = 0;

    // Set pt(mid point)
    keyLine.pt = (start_pts + end_pts)/2;

    // Set size
    keyLine.size = fabs((end_pts.x-start_pts.x) * (end_pts.y-start_pts.y));

    return keyLine;
}

void LineFeatureTracker::lineMergingTwoPhase( Mat &prev_img, Mat &cur_img, vector<LineKL> &prev_keyLine, vector<LineKL> &cur_keyLine,
                                             Mat &prev_descriptor, Mat &cur_descriptor, vector<DMatch> &good_match_vector )
{
    TicToc t_linemerging;
    int line_split = true;
    vector<uchar> temp_status;
    vector<float> temp_err;
    vector<cv::Point2f> good_points, predict_points;
    imageheight = cur_img.rows;
    Mat line_mask = Mat::zeros(imageheight, imagewidth, CV_8UC1);
    Mat temp_img = cur_img.clone();
    cvtColor(temp_img, temp_img, CV_GRAY2BGR);
    for(int i = 0; i < prev_keyLine.size(); i++)
    {
        line(line_mask, prev_keyLine.at(i).getStartPoint(), prev_keyLine.at(i).getEndPoint(), 255);
    }
    goodFeaturesToTrack(prev_img, good_points, 100, 0.01, 5, line_mask);
    calcOpticalFlowPyrLK(prev_img, cur_img, good_points, predict_points, temp_status, temp_err, cv::Size(21, 21), 3);

//    for(int i = 0; i < good_points.size(); i++)
//    {
//        circle(temp_img, good_points.at(i), 2, Scalar(0,255,0), -1);
//        line(temp_img, good_points.at(i), predict_points.at(i), Scalar(0,255,0));
//    }
//    imshow("1", temp_img);
//    waitKey(1);

    // predict lines by using calcOpticalFlowPyrLK
    vector<uchar> status, status_reduced;
    vector<float> err;
    vector<Point2f> cur_pts, forw_pts;
    int line_distribution = 2;
    int num = 0;
    vector<int> predict_idx;
    for( int i = 0; i < prev_keyLine.size(); i++)
        predict_idx.push_back(1);

    vector<DMatch>::iterator it_good_match = good_match_vector.begin();
    for ( ; it_good_match != good_match_vector.end(); it_good_match++ )
        predict_idx[it_good_match->queryIdx] = 0;

    unsigned int cur_pts_idx = 0;

    for ( int i = 0; i < prev_keyLine.size(); i++){
        if (!predict_idx[i])
            continue;
        Point2f start_pts = prev_keyLine[i].getStartPointInOctave();
        Point2f end_pts = prev_keyLine[i].getEndPointInOctave();

        if(line_distribution == 2)
        {
            cur_pts.push_back(start_pts);
            cur_pts.push_back(end_pts);
        }
        else
        {
            for(int j = 0; j < line_distribution; j++)
            {
                Point2f p = (start_pts * (line_distribution - 1 - j) + end_pts * j)/(line_distribution - 1);
                cur_pts.push_back(p);
            }
        }
        cur_pts_idx++;
    }
    if (cur_pts_idx == 0) return;

    cv::calcOpticalFlowPyrLK(prev_img, cur_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
    if(line_distribution == 2)
    {
        for( int i = 0; i < forw_pts.size(); i+=2 ){
            if ( err[i] > 15 || err[i+1] > 15 || !status[i] || !status[i+1])
                status[i] = status[i+1] = 0;
        }

        util.reduceVector4Line(cur_pts, status);
        util.reduceVector4Line(forw_pts, status);
    }
    else
    {
        status_reduced.resize(status.size()/line_distribution);

        for(int i = 0; i < forw_pts.size(); i+=line_distribution)
        {
            for(int j = 0; j < line_distribution; j++)
                if(err[i+j] > 15)
                    status[i+j] = 0;
        }

        for(int i = 0; i < forw_pts.size(); i+=line_distribution)
        {
            int one_cnt = 0;
            for(int j = 0; j < line_distribution; j++)
            {
                if(status[i + j])
                    one_cnt++;
            }

            if(one_cnt < 2)
                status_reduced[i/line_distribution] = 0;
            else
                status_reduced[i/line_distribution] = 1;
        }
    }

    // Compare predicted lines with extracted lines
    Ptr<LineBD> lineBiDes = LineBD::createBinaryDescriptor();
    vector<LineKL> predict_keylines, unextracted_lines;
    Mat predict_descriptor, unextracted_descriptor;

    Mat img1 = cur_img.clone();
    Mat img2 = cur_img.clone();
    cvtColor(img1, img1, CV_GRAY2BGR);
    cvtColor(img2, img2, CV_GRAY2BGR);


    if(line_distribution == 2)
    {
        for ( int i = 0; i < forw_pts.size(); i+=2 ){
            LineKL keyline = MakeKeyLine(forw_pts[i], forw_pts[i+1], cur_img.cols);
            keyline.class_id = i/2;
            predict_keylines.push_back(keyline);
        }
    }
    else
    {
        int line_id = 0;
        for(int i = 0; i < forw_pts.size(); i+=line_distribution)
        {
            if(!status_reduced[i/line_distribution])
                continue;

            double sum_x = 0;
            double sum_y = 0;
            double sum_x2 = 0;
            double sum_y2 = 0;
            double sum_xy = 0;

            bool isFirst = true;
            Point2f init_point;
            Point2f last_point;
            int count = 0;
            for(int j = 0; j < line_distribution; j++)
            {
                if(!status[i + j])
                    continue;
                if(isFirst)
                {
                    isFirst = false;
                    init_point = forw_pts[i + j];
                }

                sum_x += forw_pts[i + j].x;
                sum_y += forw_pts[i + j].y;
                sum_x2 += pow(forw_pts[i + j].x, 2);
                sum_y2 += pow(forw_pts[i + j].y, 2);
                sum_xy += forw_pts[i + j].x * forw_pts[i + j].y;

                last_point = forw_pts[i + j];
                count++;
            }

            double a = (count * sum_xy - sum_x * sum_y)/(count * sum_x2 - pow(sum_x,2));
            double b = sum_y/count - a * sum_x/count;
            double sigma = sqrt((sum_y2 - b * sum_y - a * sum_xy)/(count - 2));

            if(status[i] && status[i + line_distribution - 1])
                line(img1, forw_pts[i], forw_pts[i+ line_distribution - 1], Scalar(255,0,0), 2);

            //        cout << sigma << endl;
            if(sigma > 1)
            {
                status_reduced[i/line_distribution] = 0;
                continue;
            }

            double c = init_point.y + 1/a * init_point.x;
            double d = last_point.y + 1/a * last_point.x;
            double radian = atan(a);

            Point2f sp((c-b)/(a+1/a),
                       a*(c-b)/(a+1/a)+b);
            Point2f ep((d-b)/(a+1/a),
                       a*(d-b)/(a+1/a)+b);

            line(img2, sp, ep, Scalar(0,255,0), 2);

            LineKL kl = MakeKeyLine(sp, ep, cur_img.cols);

            kl.class_id = line_id;
            line_id++;
            predict_keylines.push_back(kl);
        }
//        Mat img;
//        hconcat(img1, img2, img);
//        imshow("merge", img);
//        waitKey(1);

    }

    lineBiDes->compute(cur_img, predict_keylines, predict_descriptor);

    Ptr<BinaryDescriptorMatcher> bd_match = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    vector<vector<DMatch> > matches_vector;
    bd_match->knnMatch(prev_descriptor, predict_descriptor, matches_vector, 10);
    vector<DMatch> predict_inlier_vector;
    for(int i = 0; i < matches_vector.size(); i++){
        double min_d = 1000;
        int min_idx = 0;

        for ( int j = 0; j < matches_vector[i].size(); j++){
            if ( matches_vector[i][j].queryIdx == -1 || matches_vector[i][j].trainIdx == -1 )
                continue;

            if ( matches_vector[i][j].queryIdx > (prev_keyLine.size()-1)
                || matches_vector[i][j].trainIdx > (predict_keylines.size()-1) )
                continue;

            // cout << "[" << i << ", " << j << "] : " <<  matches_vector[i][j].queryIdx << ", " << matches_vector[i][j].trainIdx << ", " << (prev_keyLine.size()-1) << ", " << (predict_keylines.size()-1) << ", " << matches_vector[i][j].distance <<  endl;

            auto query_line = prev_keyLine.at(matches_vector[i][j].queryIdx);
            auto train_line = predict_keylines.at(matches_vector[i][j].trainIdx);

            Point2f mid_p_query = (query_line.getStartPoint() + query_line.getEndPoint())/2.;
            Point2f mid_p_train = (train_line.getStartPoint() + train_line.getEndPoint())/2.;

            double dist = norm( Mat(mid_p_query), Mat(mid_p_train));

            if ( min_d > dist ){
                min_idx = j;
                min_d = dist;
            }
        }

        if (min_d < 1000){
            if ( matches_vector[i][min_idx].distance < 10 )
                predict_inlier_vector.push_back(matches_vector[i][min_idx]);
        }
    }

    Mat img_line = forw_img.clone();
    cvtColor(img_line, img_line, CV_GRAY2RGB);

    // remove duplicate points
    vector<DMatch>::iterator it_good_match_train = predict_inlier_vector.begin();
    vector<DMatch>::iterator it_good_match_query;

    for ( ; it_good_match_train != predict_inlier_vector.end()-1; ){
        if ( it_good_match_train == predict_inlier_vector.end()) break;
        bool find_duplicate = false;

        for ( it_good_match_query = it_good_match_train+1; it_good_match_query != predict_inlier_vector.end(); ){
            if ( (*it_good_match_query).trainIdx == (*it_good_match_train).trainIdx ){
                find_duplicate = true;

                if ( (*it_good_match_query).distance < (*it_good_match_train).distance )
                    predict_inlier_vector.erase(it_good_match_train);
                else{
                    predict_inlier_vector.erase(it_good_match_query);
                    it_good_match_train++;
                }

                break;
            }
            else
                it_good_match_query++;
        }

        if (!find_duplicate) it_good_match_train++;
    }

    unsigned int correct = 0, uncorrect = 0;
    vector<DMatch>::iterator it_predict_inlier = predict_inlier_vector.begin();
    int class_id = cur_keyLine.size()+1;

    for ( ; it_predict_inlier != predict_inlier_vector.end(); ){
        if ( FindMatchedLine(prev_keyLine.at(it_predict_inlier->queryIdx),
                            predict_keylines.at(it_predict_inlier->trainIdx), 20, 50, 0.17) ){

            auto start_predict_pts = predict_keylines.at(it_predict_inlier->trainIdx).getStartPointInOctave();
            auto end_predict_pts = predict_keylines.at(it_predict_inlier->trainIdx).getEndPointInOctave();

            double length = norm(Mat(start_predict_pts), Mat(end_predict_pts));

            if (length > 10)
            {
                // If the linew have the same class_id, it can't be matched.
                // Therefore, when new line is inserted the vector, it should be changed the class_id.
                m_matched_keyLines.push_back(predict_keylines.at(it_predict_inlier->trainIdx));
                (m_matched_keyLines.end()-1)->class_id = class_id;
                vconcat(m_matched_descriptor, predict_descriptor.row(it_predict_inlier->trainIdx), m_matched_descriptor);

                cur_keyLine.push_back(predict_keylines.at(it_predict_inlier->trainIdx));
                (cur_keyLine.end()-1)->class_id = class_id;
                vconcat(cur_descriptor, predict_descriptor.row(it_predict_inlier->trainIdx), cur_descriptor);

                DMatch tmp_match;
                tmp_match.queryIdx = it_predict_inlier->queryIdx;
                tmp_match.trainIdx = cur_keyLine.size()-1;
                good_match_vector.push_back(tmp_match);

                cv::Point2f start_pts = predict_keylines.at(it_predict_inlier->trainIdx).getStartPoint();
                cv::Point2f end_pts = predict_keylines.at(it_predict_inlier->trainIdx).getEndPoint();

                circle(img_line, start_pts, 2, Scalar(255, 0, 0), -1);
                circle(img_line, end_pts, 2, Scalar(255, 0, 0), -1);
                line(img_line, start_pts, end_pts, Scalar(0,255,0), 1);

//                imshow("img_line", img_line);
//                waitKey(1);
                class_id++;

                it_predict_inlier++;
                correct++;
            }
            else{
                predict_inlier_vector.erase(it_predict_inlier);
                uncorrect++;
            }

        }
        else{
            predict_inlier_vector.erase(it_predict_inlier);
            uncorrect++;
        }
    }
//    cout << t_linemerging.toc() << endl;
}

void LineFeatureTracker::lineExtraction( Mat &cur_img, vector<LineKL> &keyLine, Mat &descriptor)
{
//    line_descriptor::BinaryDescriptor::EDLineDetector;
    keyLine.clear();
    Ptr<LineBD> lineBiDes = LineBD::createBinaryDescriptor();
    Ptr<line_descriptor::LSDDetector> lineLSD = line_descriptor::LSDDetector::createLSDDetector();
    Mat keyLine_mask = Mat::ones(forw_img.size(), CV_8UC1);
//    lineLSD->detect(cur_img, keyLine, 2, 2, keyLine_mask);
    TicToc t_r;
//    lineBiDes->detect(cur_img, keyLine);
//    double t1 = t_r.toc();
    upm::Segments segs = elsed.detect(forw_img);
//    cout<< t_r.toc() << endl;
//    cout << t1 << " , " << t2 << endl;
//    Mat tmp_img1, tmp_img2;
//    cvtColor(forw_img, tmp_img1, CV_GRAY2BGR);
//    cvtColor(forw_img, tmp_img2, CV_GRAY2BGR);
//    for(int i = 0; i < forw_keyLine.size(); i++)
//        line(tmp_img1, forw_keyLine[i].getStartPoint(), forw_keyLine[i].getEndPoint(), Scalar(0,255,0), 2);
//    for(int i = 0; i < segs.size(); i++)
//    {
//        upm::Segment seg = segs[i];
//        line(tmp_img2, cv::Point2f(seg[0], seg[1]), cv::Point2f(seg[2], seg[3]), Scalar(0,255,0), 2);
//    }
//    Mat compare;
//    hconcat(tmp_img1, tmp_img2, compare);
//    imshow("1", compare);
//    waitKey(1);

    int line_id = 0;
    for(unsigned int i = 0; i < segs.size(); i++)
    {
        //用opencv的keyline记录每根线的信息
        LineKL kl = MakeKeyLine(cv::Point2f(segs[i][0], segs[i][1]), cv::Point2f(segs[i][2], segs[i][3]), cur_img.cols);
//        if(kl.lineLength < 40)
//            continue;
        kl.class_id = line_id;
        line_id++;
        keyLine.push_back(kl);

    }
    if(keyLine.size() > 0)
       lineBiDes->compute(cur_img, keyLine, descriptor);

    // for(int row = 0; row < keyLine_mask.rows-1; row++){
    //   uchar* p = keyLine_mask.ptr<uchar>(row); //pointer p points to the first place of each row
    //   for(int col = 0; col < keyLine_mask.cols/2; col++)
    //     p[col] = 0;
    // }

    /* delete undesired KeyLines, according to input mask and filtering by lenth of a line*/
    vector<LineKL>::iterator it_keyLine = keyLine.begin();
    int idx = 0;


//    Mat img1 = cur_img.clone();
//    Mat img2 = cur_img.clone();

//    cvtColor(img1, img1, CV_GRAY2BGR);
//    cvtColor(img2, img2, CV_GRAY2BGR);
//    for(auto &it : keyLine)
//        line(img1, it.getStartPoint(), it.getEndPoint(),Scalar(rand()%255,rand()%255,rand()%255), 2);

//    bool endflag = false;
//    for(int i = 0; i < int(keyLine.size()); i++)
//    {
//        KeyLine kl1 = keyLine[i];
//        for(int j = i + 1 ; j < int(keyLine.size()); j++)
//        {
//            KeyLine kl2 = keyLine[j];
//            double a = kl1.angle;
//            double b = kl1.startPointY - (kl1.endPointY - kl1.startPointY)/(kl1.endPointX - kl1.startPointX)*kl1.startPointX;
//            double c = kl2.angle;
//            double d = kl2.startPointY - (kl2.endPointY - kl2.startPointY)/(kl2.endPointX - kl2.startPointX)*kl2.startPointX;

//            Point2f p1(kl1.startPointX, kl1.startPointY);
//            Point2f p2(kl1.endPointX, kl1.endPointY);
//            Point2f p3(kl2.startPointX, kl2.startPointY);
//            Point2f p4(kl2.endPointX, kl2.endPointY);

//            Vector2d v1(kl1.startPointX, kl1.startPointY);
//            Vector2d v2(kl1.endPointX, kl1.endPointY);
//            Vector2d v3(kl2.startPointX, kl2.startPointY);
//            Vector2d v4(kl2.endPointX, kl2.endPointY);

//            double max_dist = max((v1-v3).norm(), max((v1-v4).norm(), max((v2-v3).norm(), max((v2-v4).norm(), max((v1-v2).norm(), (v3-v4).norm())))));
//            double min_dist = min((v1-v3).norm(), min((v1-v4).norm(), min((v2-v3).norm(), (v2-v4).norm())));

//            LineKL kl;
//            if(max_dist == (v1-v3).norm())
//            {
//                kl = MakeKeyLine(p1, p3, cur_img.cols);
//                //                cout << "1, 3" << endl;
//            }
//            else if(max_dist == (v1-v4).norm())
//            {
//                kl = MakeKeyLine(p1, p4, cur_img.cols);
//                //                cout << "1, 4" << endl;
//            }
//            else if(max_dist == (v2-v3).norm())
//            {
//                kl = MakeKeyLine(p2, p3, cur_img.cols);
//                //                cout << "1, 3" << endl;
//            }
//            else if(max_dist == (v2-v4).norm())
//            {
//                kl = MakeKeyLine(p2, p4, cur_img.cols);
//                //                cout << "1, 3" << endl;
//            }
//            else if(max_dist == (v1-v2).norm())
//            {
//                kl = kl1;
//                //                cout << "1, 2" << endl;
//            }
//            else if(max_dist == (v3-v4).norm())
//            {
//                kl = kl2;
//                //                cout << "3, 4" << endl;
//            }

//            if(abs(a-c) < 1 * M_PI/180 || abs(abs(a-c) - M_PI) < 1 * M_PI/180)
//            {
//                if(abs(b-d) < 1 && min_dist < 10)
//                {
////                    line(image, p1, p2, Scalar(255, 0, 0), 3);
////                    line(image, p3, p4, Scalar(0, 0, 255), 2);
////                    line(image, kl.getStartPoint(), kl.getEndPoint(), Scalar(0, 255, 0), 1);

//                    kl.class_id = kl1.class_id;
//                    keyLine[i] = kl;

//                    keyLine.erase(keyLine.begin() + j);
//                    util.DeleteCVMatRow(descriptor, j);

//                    endflag = true;
//                    break;
//                }
//            }
//        }
//        if (endflag)
//        {
//            endflag = false;
//            continue;
//        }
//    }


//    for (;it_keyLine != keyLine.end();){
//        double d = norm( Mat((*it_keyLine).getStartPoint()), Mat((*it_keyLine).getEndPoint()) );
//        KeyLine& kl = *it_keyLine;

//        ///// NEED TO CHECK --> min? (KWANG YIK)
//        float srt_x_at_max_y =  ((keyLine_mask.rows - 1) - kl.startPointY)*(kl.endPointX - kl.startPointX)/(kl.endPointY - kl.startPointY) + kl.startPointX;
//        float end_x_at_max_y =  ((keyLine_mask.rows - 1) - kl.endPointY)*(kl.endPointX - kl.startPointX)/(kl.endPointY - kl.startPointY) + kl.endPointX;
//        float srt_y_at_max_x =  ((keyLine_mask.cols - 1) - kl.startPointX)*(kl.endPointY - kl.startPointY)/(kl.endPointX - kl.startPointX) + kl.startPointY;
//        float end_y_at_max_x =  ((keyLine_mask.cols - 1) - kl.endPointX)*(kl.endPointY - kl.startPointY)/(kl.endPointX - kl.startPointX) + kl.endPointY;

//        if((int)kl.startPointX > keyLine_mask.cols - 1)
//        {
//            kl.startPointX = srt_x_at_max_y;
//        }
//        if((int)kl.startPointY > keyLine_mask.rows - 1)
//        {
//            kl.startPointY = srt_y_at_max_x;
//        }
//        if((int)kl.endPointX > keyLine_mask.cols - 1)
//        {
//            kl.endPointX = end_x_at_max_y;
//        }
//        if((int)kl.endPointY > keyLine_mask.rows - 1)
//        {
//            kl.endPointY = end_y_at_max_x;
//        }
//        //due to imprecise floating point scaling in the pyramid a little overflow can occur in line coordinates,
//        //especially on big images. It will be fixed here
//        //kl.startPointX = (float)std::min((int)kl.startPointX, keyLine_mask.cols - 1);
//        //kl.startPointY = (float)std::min((int)kl.startPointY, keyLine_mask.rows - 1);
//        //kl.endPointX = (float)std::min((int)kl.endPointX, keyLine_mask.cols - 1);
//        //kl.endPointY = (float)std::min((int)kl.endPointY, keyLine_mask.rows - 1);

//        if((keyLine_mask.at < uchar > ( (int) kl.startPointY, (int) kl.startPointX ) == 0 ||
//             keyLine_mask.at < uchar > ( (int) kl.endPointY, (int) kl.endPointX ) == 0)
//            || kl.lineLength < 50
//                || kl.octave != 0){
//            keyLine.erase(it_keyLine);
//            util.DeleteCVMatRow(descriptor, idx);
//        }
//        else{
//            it_keyLine++;
//            idx++;
//        }
//    }
}

void LineFeatureTracker::normalizePoints(){
    curr_start_un_pts.clear();
    curr_end_un_pts.clear();

    for (unsigned int i = 0; i < curr_start_pts.size(); i++)
    {
        Eigen::Vector2d start_a(curr_start_pts[i].x, curr_start_pts[i].y);
        Eigen::Vector3d start_b;
        Eigen::Vector2d end_a(curr_end_pts[i].x, curr_end_pts[i].y);
        Eigen::Vector3d end_b;

        pinhole_camera->liftProjective4line(start_a, start_b);
        pinhole_camera->liftProjective4line(end_a, end_b);

        curr_start_un_pts.push_back(Point2f(start_b.x() / start_b.z(), start_b.y() / start_b.z()));
        curr_end_un_pts.push_back(Point2f(end_b.x() / end_b.z(), end_b.y() / end_b.z()));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }

}

void LineFeatureTracker::imageUndistortion(Mat &_img, Mat &_out_undistort_img)
{
    // camera type selected image undistortion
    Mat original_image = _img;
    Mat undistorted_image;

    float resize_rate = 1.0;
    Camera_Matrix.at<float>(0,0) = PROJ_FX * resize_rate;
    Camera_Matrix.at<float>(1,1) = PROJ_FY * resize_rate;
    Camera_Matrix.at<float>(0,2) = PROJ_CX * resize_rate;
    Camera_Matrix.at<float>(1,2) = PROJ_CY * resize_rate;
    Camera_Matrix.at<float>(2,2) = 1.0;

    Discotrion_Coefficients.at<float>(0,0) = DIST_K1;
    Discotrion_Coefficients.at<float>(0,1) = DIST_K2;
    Discotrion_Coefficients.at<float>(0,2) = DIST_P1;
    Discotrion_Coefficients.at<float>(0,3) = DIST_P2;

    undistort(original_image, undistorted_image, Camera_Matrix, Discotrion_Coefficients);

    Mat merge_img;
    vconcat(original_image, undistorted_image, merge_img); //垂直合并
    cv_bridge::CvImage out_img_msg;
    out_img_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
    out_img_msg.image = merge_img;

    _out_undistort_img = undistorted_image;
}

void LineFeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    pinhole_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file, true);

    if(DIST_K1 > 0)
    {
        PinholeCamera::Parameters new_parameters = pinhole_camera->getParameters();
        new_parameters.cx() -= COL_MARGIN;
        new_parameters.cy() -= ROW_MARGIN;
        pinhole_camera->setParameters(new_parameters);
    }
}
//TODO define updateId
bool LineFeatureTracker::updateID(unsigned int i)
{
    if(i < ids.size())
    {
        if(ids[i] == -1)
        {
            ids[i] = n_id++;
        }
        return true;
    }
    else
        return false;
}

void LineFeatureTracker::CannyDetection(Mat &src, vector<line_descriptor::KeyLine> &keylines)
{
    keylines.clear();
    float r, c;
    imageheight=src.rows; imagewidth=src.cols;

    Mat canny = src.clone();
    Canny(src, canny, 50, 50, 3);

    vector<Point2f> points;
    vector<line_descriptor::KeyLine> kls, kls_tmp, kls_tmp2;

    for(int i=0; i<src.rows;i++) {
        for(int j=0; j<src.cols;j++) {
            if( i < 5 || i > src.rows-5 || j < 5 || j > src.cols - 5)
                canny.at<unsigned char>(i,j) = 0;
        }
    }

    line_descriptor::KeyLine kl, kl1, kl2;

    for ( r = 0; r < src.rows; r++ ) {
        for ( c = 0; c < src.cols; c++ ) {
            // Find seeds - skip for non-seeds
            if ( canny.at<unsigned char>(r,c) == 0 )
                continue;

            // Found seeds
            Point2f pt;
            pt.x = c;
            pt.y = r;

            points.push_back(pt);
            canny.at<unsigned char>(pt.y, pt.x) = 0;

            int direction = 0;
            int step = 0;
            while (getPointChain( canny, pt, &pt, direction, step)) {
                points.push_back(pt);
                step++;
                canny.at<unsigned char>(pt.y, pt.x) = 0;
            }

            if ( points.size() < (unsigned int)threshold_length + 1 ) {
                points.clear();
                continue;
            }

            extractSegments( &points, &kls);

            if ( kls.size() == 0 ) {
                points.clear();
                continue;
            }

            for ( int i = 0; i < (int)kls.size(); i++ ) {
                kl = kls.at(i);
                float length = sqrt(pow((kl.startPointX - kl.endPointX),2) + pow((kl.startPointY - kl.endPointY),2));
                if(length < threshold_length) continue;
                if( (kl.startPointX <= 5.0f && kl.endPointX <= 5.0f)
                    || (kl.startPointY <= 5.0f && kl.endPointY <= 5.0f)
                    || (kl.startPointX >= imagewidth - 5.0f && kl.endPointX >= imagewidth - 5.0f)
                    || (kl.startPointY >= imageheight - 5.0f && kl.endPointY >= imageheight - 5.0f) )
                    continue;

                kls_tmp.push_back(kl);
            }

            points.clear();
            kls.clear();
        }
    }

    bool is_merged = false;
    int ith = kls_tmp.size() - 1;
    int jth = ith - 1;
    while(false)
    {
        kl1 = kls_tmp[ith];
        kl2 = kls_tmp[jth];
        is_merged = mergeSegments(&kl1, &kl2, &kl2);

        if(is_merged)
        {
            additionalOperationsOnSegments(src, &kl2);
            vector<line_descriptor::KeyLine>::iterator it = kls_tmp.begin() + ith;
            *it = kl2;
            kls_tmp.erase(kls_tmp.begin()+jth);
            ith--;
            jth = ith - 1;
        }
        else
            jth--;
        if(jth < 0)
        {
            ith--;
            jth = ith - 1;
        }
        if(ith == 1 && jth == 0)
            break;
    }

    keylines = kls_tmp;
}

bool LineFeatureTracker::getPointChain( const Mat & img, const Point2f pt, Point2f * chained_pt,
                                       int & direction, int step )
{
    float ri, ci;
    int indices[8][2]={ {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1},{-1,0}, {-1,1}, {0,1} };

    for ( int i = 0; i < 8; i++ ) {
        ci = pt.x + indices[i][1];
        ri = pt.y + indices[i][0];

        if ( ri < 0 || ri == img.rows || ci < 0 || ci == img.cols )
            continue;

        if ( img.at<unsigned char>(ri, ci) == 0 )
            continue;

        if(step == 0) {
            chained_pt->x = ci;
            chained_pt->y = ri;
            direction = i;
            return true;
        } else {
            if(abs(i-direction) <= 2 || abs(i-direction) >= 6)
            {
                chained_pt->x = ci;
                chained_pt->y = ri;
                direction = i;
                return true;
            } else
                continue;
        }
    }
    return false;
}

void LineFeatureTracker::extractSegments( vector<Point2f> * points, vector<line_descriptor::KeyLine> * kls)
{
    bool is_line;

    int i, j;
    line_descriptor::KeyLine kl;
    Point2f ps, pe, pt;

    vector<Point2f> l_points;

    int total = points->size();

    for ( i = 0; i + threshold_length < total; i++ ) {
        ps = points->at(i);
        pe = points->at(i + threshold_length);

        double a[] = { (double)ps.x, (double)ps.y, 1 };
        double b[] = { (double)pe.x, (double)pe.y, 1 };
        double c[3], d[3];

        Mat p1 = Mat(3, 1, CV_64FC1, a).clone();
        Mat p2 = Mat(3, 1, CV_64FC1, b).clone();
        Mat p = Mat(3, 1, CV_64FC1, c).clone();
        Mat l = Mat(3, 1, CV_64FC1, d).clone();
        l = p1.cross(p2);

        is_line = true;

        l_points.clear();
        l_points.push_back(ps);

        for ( j = 1; j < threshold_length; j++ ) {
            pt.x = points->at(i+j).x;
            pt.y = points->at(i+j).y;

            p.at<double>(0,0) = (double)pt.x;
            p.at<double>(1,0) = (double)pt.y;
            p.at<double>(2,0) = 1.0;

            double dist = distPointLine( p, l );

            if ( fabs( dist ) > threshold_dist ) {
                is_line = false;
                break;
            }
            l_points.push_back(pt);
        }

        // Line check fail, test next point
        if ( is_line == false )
            continue;

        l_points.push_back(pe);

        Vec4f line;
        fitLine( Mat(l_points), line, CV_DIST_L2, 0, 0.01, 0.01);
        a[0] = line[2];
        a[1] = line[3];
        b[0] = line[2] + line[0];
        b[1] = line[3] + line[1];

        p1 = Mat(3, 1, CV_64FC1, a).clone();
        p2 = Mat(3, 1, CV_64FC1, b).clone();

        l = p1.cross(p2);

        incidentPoint( &ps, l );

        // Extending line
        for ( j = threshold_length + 1; i + j < total; j++ ) {
            pt.x = points->at(i+j).x;
            pt.y = points->at(i+j).y;

            p.at<double>(0,0) = (double)pt.x;
            p.at<double>(1,0) = (double)pt.y;
            p.at<double>(2,0) = 1.0;

            double dist = distPointLine( p, l );

            if ( fabs( dist ) > threshold_dist ) {
                j--;
                break;
            }

            pe = pt;
            l_points.push_back(pt);
        }
        fitLine( Mat(l_points), line, CV_DIST_L2, 0, 0.01, 0.01);
        a[0] = line[2];
        a[1] = line[3];
        b[0] = line[2] + line[0];
        b[1] = line[3] + line[1];

        p1 = Mat(3, 1, CV_64FC1, a).clone();
        p2 = Mat(3, 1, CV_64FC1, b).clone();

        l = p1.cross(p2);

        Point2f e1, e2;
        e1.x = ps.x;
        e1.y = ps.y;
        e2.x = pe.x;
        e2.y = pe.y;

        incidentPoint( &e1, l );
        incidentPoint( &e2, l );

        i = i + j;
        kl = MakeKeyLine(e1, e2, imagewidth);
        kl.class_id = 0;

        kls->push_back(kl);
    }
}

double LineFeatureTracker::distPointLine( const Mat & p, Mat & l )
{
    double x, y, w;

    x = l.at<double>(0,0);
    y = l.at<double>(1,0);

    w = sqrt(x*x+y*y);

    l.at<double>(0,0) = x  / w;
    l.at<double>(1,0) = y  / w;
    l.at<double>(2,0) = l.at<double>(2,0)  / w;

    return l.dot(p);
}

template<class tType>
void LineFeatureTracker::incidentPoint( tType * pt, Mat & l )
{
    double a[] = { (double)pt->x, (double)pt->y, 1.0 };
    double b[] = { l.at<double>(0,0), l.at<double>(1,0), 0.0 };
    double c[3];

    Mat xk = Mat(3, 1, CV_64FC1, a).clone();
    Mat lh = Mat(3, 1, CV_64FC1, b).clone();
    Mat lk = Mat(3, 1, CV_64FC1, c).clone();

    lk = xk.cross(lh);
    xk = lk.cross(l);

    double s = 1.0 / xk.at<double>(2,0);
    xk.convertTo(xk, -1, s);

    pt->x = (float)xk.at<double>(0,0) < 0.0f ? 0.0f : (float)xk.at<double>(0,0)
                                                                >= (imagewidth - 1.0f) ? (imagewidth - 1.0f) : (float)xk.at<double>(0,0);
    pt->y = (float)xk.at<double>(1,0) < 0.0f ? 0.0f : (float)xk.at<double>(1,0)
                                                                >= (imageheight - 1.0f) ? (imageheight - 1.0f) : (float)xk.at<double>(1,0);

}

void LineFeatureTracker::additionalOperationsOnSegments(Mat & src, line_descriptor::KeyLine * kl)
{
    if(kl->startPointX == 0.0f && kl->startPointY == 0.0f && kl->endPointX == 0.0f && kl->endPointY == 0.0f)
        return;

    double ang = (double)kl->angle;

    Point2f start = Point2f(kl->startPointX, kl->startPointY);
    Point2f end = Point2f(kl->endPointX, kl->endPointY);

    double dx = 0.0, dy = 0.0;
    dx = (double) end.x - (double) start.x;
    dy = (double) end.y - (double) start.y;

    int num_points = 10;
    Point2f *points = new Point2f[num_points];

    points[0] = start;
    points[num_points - 1] = end;
    for (int i = 0; i < num_points; i++) {
        if (i == 0 || i == num_points - 1)
            continue;
        points[i].x = points[0].x + (dx / double(num_points - 1) * (double) i);
        points[i].y = points[0].y + (dy / double(num_points - 1) * (double) i);
    }

    Point2f *points_right = new Point2f[num_points];
    Point2f *points_left = new Point2f[num_points];
    double gap = 1.0;

    for(int i = 0; i < num_points; i++) {
        points_right[i].x = cvRound(points[i].x + gap*cos(90.0 * CV_PI / 180.0 + ang));
        points_right[i].y = cvRound(points[i].y + gap*sin(90.0 * CV_PI / 180.0 + ang));
        points_left[i].x = cvRound(points[i].x - gap*cos(90.0 * CV_PI / 180.0 + ang));
        points_left[i].y = cvRound(points[i].y - gap*sin(90.0 * CV_PI / 180.0 + ang));
        pointInboardTest(src, &points_right[i]);
        pointInboardTest(src, &points_left[i]);
    }

    delete[] points; delete[] points_right; delete[] points_left;

    return;
}

void LineFeatureTracker::pointInboardTest(Mat & src, Point2f * pt)
{
    pt->x = pt->x <= 5.0f ? 5.0f : pt->x >= src.cols - 5.0f ? src.cols - 5.0f : pt->x;
    pt->y = pt->y <= 5.0f ? 5.0f : pt->y >= src.rows - 5.0f ? src.rows - 5.0f : pt->y;
}

bool LineFeatureTracker::mergeSegments( line_descriptor::KeyLine* kl1, line_descriptor::KeyLine* kl2, line_descriptor::KeyLine* kl_merged )
{
    double o[] = { 0.0, 0.0, 1.0 };
    double a[] = { 0.0, 0.0, 1.0 };
    double b[] = { 0.0, 0.0, 1.0 };
    double c[3];

    double seg1_x1 = kl1->startPointX;
    double seg1_y1 = kl1->startPointY;
    double seg1_x2 = kl1->endPointX;
    double seg1_y2 = kl1->endPointY;
    double seg2_x1 = kl2->startPointX;
    double seg2_y1 = kl2->startPointY;
    double seg2_x2 = kl2->endPointX;
    double seg2_y2 = kl2->endPointY;

    o[0] = ( seg2_x1 + seg2_x2 ) / 2.0;
    o[1] = ( seg2_y1 + seg2_y2 ) / 2.0;

    a[0] = seg1_x1;
    a[1] = seg1_y1;
    b[0] = seg1_x2;
    b[1] = seg1_y2;

    Mat ori = Mat(3, 1, CV_64FC1, o).clone();
    Mat p1 = Mat(3, 1, CV_64FC1, a).clone();
    Mat p2 = Mat(3, 1, CV_64FC1, b).clone();
    Mat l1 = Mat(3, 1, CV_64FC1, c).clone();

    l1 = p1.cross(p2);

    Point2f seg1mid, seg2mid;
    seg1mid.x = (seg1_x1 + seg1_x2) /2.0f;
    seg1mid.y = (seg1_y1 + seg1_y2) /2.0f;
    seg2mid.x = (seg2_x1 + seg2_x2) /2.0f;
    seg2mid.y = (seg2_y1 + seg2_y2) /2.0f;

    double seg1len, seg2len;
    seg1len = sqrt((seg1_x1 - seg1_x2)*(seg1_x1 - seg1_x2)+(seg1_y1 - seg1_y2)*(seg1_y1 - seg1_y2));
    seg2len = sqrt((seg2_x1 - seg2_x2)*(seg2_x1 - seg2_x2)+(seg2_y1 - seg2_y2)*(seg2_y1 - seg2_y2));

    double middist = sqrt((seg1mid.x - seg2mid.x)*(seg1mid.x - seg2mid.x) + (seg1mid.y - seg2mid.y)*(seg1mid.y - seg2mid.y));

    float angdiff = kl1->angle - kl2->angle;
    angdiff = fabs(angdiff);

    double dist = distPointLine( ori, l1 );

    if ( fabs( dist ) <= threshold_dist
        && middist <= seg1len / 2.0 + seg2len / 2.0
        && (angdiff <= CV_PI / 180.0f * 1.0f || abs(angdiff - CV_PI) <= CV_PI / 180.0f * 1.0f)) {
        mergeLines(kl1, kl2, kl2);
        return true;
    } else {
        return false;
    }
}

void LineFeatureTracker::mergeLines(line_descriptor::KeyLine * kl1, line_descriptor::KeyLine * kl2, line_descriptor::KeyLine * kl_merged)
{
    double seg1_x1 = kl1->startPointX;
    double seg1_y1 = kl1->startPointY;
    double seg1_x2 = kl1->endPointX;
    double seg1_y2 = kl1->endPointY;
    double seg2_x1 = kl2->startPointX;
    double seg2_y1 = kl2->startPointY;
    double seg2_x2 = kl2->endPointX;
    double seg2_y2 = kl2->endPointY;

    double xg = 0.0, yg = 0.0;
    double delta1x = 0.0, delta1y = 0.0, delta2x = 0.0, delta2y = 0.0;
    float ax = 0, bx = 0, cx = 0, dx = 0;
    float ay = 0, by = 0, cy = 0, dy = 0;
    double li = 0.0, lj = 0.0;
    double thi = 0.0, thj = 0.0, thr = 0.0;
    double axg = 0.0, bxg = 0.0, cxg = 0.0, dxg = 0.0, delta1xg = 0.0, delta2xg = 0.0;

    ax = seg1_x1;
    ay = seg1_y1;
    bx = seg1_x2;
    by = seg1_y2;
    cx = seg2_x1;
    cy = seg2_y1;
    dx = seg2_x2;
    dy = seg2_y2;

    float dlix = (bx - ax);
    float dliy = (by - ay);
    float dljx = (dx - cx);
    float dljy = (dy - cy);

    li = sqrt((double) (dlix * dlix) + (double) (dliy * dliy));
    lj = sqrt((double) (dljx * dljx) + (double) (dljy * dljy));

    xg = (li * (double) (ax + bx) + lj * (double) (cx + dx))
         / (double) (2.0 * (li + lj));
    yg = (li * (double) (ay + by) + lj * (double) (cy + dy))
         / (double) (2.0 * (li + lj));

    if(dlix == 0.0f) thi = CV_PI / 2.0;
    else thi = atan(dliy / dlix);

    if(dljx == 0.0f) thj = CV_PI / 2.0;
    else thj = atan(dljy / dljx);

    if (fabs(thi - thj) <= CV_PI / 2.0)
    {
        thr = (li * thi + lj * thj) / (li + lj);
    }
    else
    {
        double tmp = thj - CV_PI * (thj / fabs(thj));
        thr = li * thi + lj * tmp;
        thr /= (li + lj);
    }

    axg = ((double) ay - yg) * sin(thr) + ((double) ax - xg) * cos(thr);

    bxg = ((double) by - yg) * sin(thr) + ((double) bx - xg) * cos(thr);

    cxg = ((double) cy - yg) * sin(thr) + ((double) cx - xg) * cos(thr);

    dxg = ((double) dy - yg) * sin(thr) + ((double) dx - xg) * cos(thr);

    delta1xg = min(axg,min(bxg,min(cxg,dxg)));
    delta2xg = max(axg,max(bxg,max(cxg,dxg)));

    delta1x = delta1xg * cos(thr) + xg;
    delta1y = delta1xg * sin(thr) + yg;
    delta2x = delta2xg * cos(thr) + xg;
    delta2y = delta2xg * sin(thr) + yg;

    line_descriptor::KeyLine kl_tmp = MakeKeyLine(Point2f(delta1x, delta1y), Point2f(delta2x, delta2y), imagewidth);
    kl_tmp.class_id = kl2->class_id;
    *kl_merged = kl_tmp;
}

void LineFeatureTracker::OpticalFlowExtraction(Mat &prev_img, Mat &cur_img,
                                               vector<LineKL> &prev_keyLine, vector<LineKL> &cur_keyLine,
                                               Mat &prev_descriptor, Mat &cur_descriptor)
{
    vector<uchar> status;
    vector<uchar> status_reduced;
    vector<float> err;
    vector<Point2f> cur_pts, forw_pts;
    int num = 0;
    int line_distribution = 2;
    Mat canny_img;
    Canny(cur_img, canny_img, 50, 150, 3);
    cur_keyLine.clear();

    for(int i = 0; i < prev_keyLine.size(); i++)
    {
        Point2f start_pts = prev_keyLine[i].getStartPointInOctave();
        Point2f end_pts = prev_keyLine[i].getEndPointInOctave();
        for(int j = 0; j < line_distribution; j++)
        {
            Point2f p = (start_pts * (line_distribution - 1 - j) + end_pts * j)/(line_distribution - 1);
            cur_pts.push_back(p);
        }
    }

    cv::calcOpticalFlowPyrLK(prev_img, cur_img, cur_pts, forw_pts, status, err, cv::Size(21,21), 3);

    status_reduced.resize(status.size()/line_distribution);

    for( int i = 0; i < forw_pts.size(); i+=line_distribution ){
        bool isBad = false;
        for(int j = 0; j < line_distribution; j++)
        {
            if(!util.inBorder(forw_pts[i + j]) || !status[i + j])
            {
                isBad = true;
                break;
            }
        }

        if(isBad)
            status_reduced[i/line_distribution] = 0;
        else
            status_reduced[i/line_distribution] = 1;
    }

    for(int i = 0; i < forw_pts.size(); i+=line_distribution)
    {
        if(!status_reduced[i/line_distribution])
            continue;

        double sum_x = 0;
        double sum_y = 0;
        double sum_x2 = 0;
        double sum_y2 = 0;
        double sum_xy = 0;

        for(int j = 0; j < line_distribution; j++)
        {
            sum_x += forw_pts[i + j].x;
            sum_y += forw_pts[i + j].y;
            sum_x2 += pow(forw_pts[i + j].x, 2);
            sum_y2 += pow(forw_pts[i + j].y, 2);
            sum_xy += forw_pts[i + j].x * forw_pts[i + j].y;
        }

        double a = (line_distribution * sum_xy - sum_x * sum_y)/(line_distribution * sum_x2 - pow(sum_x,2));
        double b = sum_y/line_distribution - a * sum_x/line_distribution;
        double length = norm(forw_pts[i] - forw_pts[i + line_distribution - 1]);
        double radian = atan(a);
        double sigma = sqrt((sum_y2 - b * sum_y - a * sum_xy)/(line_distribution - 2));

        if(sigma > 1)
        {
            status_reduced[i/line_distribution] = 0;
            continue;
        }

        Point2f sp(sum_x/line_distribution - length/2 * cos(radian),
                   sum_y/line_distribution - length/2 * sin(radian));
        Point2f ep(sum_x/line_distribution + length/2 * cos(radian),
                   sum_y/line_distribution + length/2 * sin(radian));

        LineKL kl = MakeKeyLine(sp, ep, cur_img.cols);
        kl.class_id = i/line_distribution;
        cur_keyLine.push_back(kl);
    }

    num = 0;
    for(int i = 0; i < cur_pts.size(); i+=line_distribution)
    {
        if(status_reduced[i / line_distribution])
        {
            for(int j = 0; j < line_distribution; j++)
            {
                cur_pts[num] = cur_pts[i + j];
                num++;
            }
        }
    }
    cur_pts.resize(num);

    num = 0;
    for(int i = 0; i < forw_pts.size(); i+=line_distribution)
    {
        if(status_reduced[i / line_distribution])
        {
            for(int j = 0; j < line_distribution; j++)
            {
                forw_pts[num] = forw_pts[i + j];
                num++;
            }
        }
    }
    forw_pts.resize(num);

    util.reduceVector(ids, status_reduced);
    util.reduceVector(track_cnt, status_reduced);

    for (auto &n : track_cnt)
        n++;

    vector<LineKL> newKLs;
    int idx = 0;
    if(cur_keyLine.size() < 150)
    {
        Mat keyLine_mask;
        Ptr<line_descriptor::LSDDetector> lineLSD = line_descriptor::LSDDetector::createLSDDetector();
        Ptr<LineBD> lineBiDes = LineBD::createBinaryDescriptor();

        //        lineBiDes->detect(cur_img, newKLs);
        //        lineLSD->detect(cur_img, newKLs, 10, 1, keyLine_mask);
        CannyDetection(cur_img, newKLs);

        bool endflag = false;
        for(int i = 0; i < newKLs.size(); i++)
        {
            LineKL kl1 = newKLs[i];
            for(int j = 0; j < cur_keyLine.size(); j++)
            {
                LineKL kl2 = cur_keyLine[j];

                double a = kl1.angle;
                double b = kl1.startPointY - (kl1.endPointY - kl1.startPointY)/(kl1.endPointX - kl1.startPointX)*kl1.startPointX;
                double c = kl2.angle;
                double d = kl2.startPointY - (kl2.endPointY - kl2.startPointY)/(kl2.endPointX - kl2.startPointX)*kl2.startPointX;

                Point2f p1(kl1.startPointX, kl1.startPointY);
                Point2f p2(kl1.endPointX, kl1.endPointY);
                Point2f p3(kl2.startPointX, kl2.startPointY);
                Point2f p4(kl2.endPointX, kl2.endPointY);

                Vector2d v1(kl1.startPointX, kl1.startPointY);
                Vector2d v2(kl1.endPointX, kl1.endPointY);
                Vector2d v3(kl2.startPointX, kl2.startPointY);
                Vector2d v4(kl2.endPointX, kl2.endPointY);

                double max_dist = max((v1-v3).norm(), max((v1-v4).norm(), max((v2-v3).norm(), max((v2-v4).norm(), max((v1-v2).norm(), (v3-v4).norm())))));
                double min_dist = min((v1-v3).norm(), min((v1-v4).norm(), min((v2-v3).norm(), (v2-v4).norm())));

                LineKL kl;
                if(max_dist == (v1-v3).norm())
                {
                    kl = MakeKeyLine(p1, p3, cur_img.cols);
                    //                cout << "1, 3" << endl;
                }
                else if(max_dist == (v1-v4).norm())
                {
                    kl = MakeKeyLine(p1, p4, cur_img.cols);
                    //                cout << "1, 4" << endl;
                }
                else if(max_dist == (v2-v3).norm())
                {
                    kl = MakeKeyLine(p2, p3, cur_img.cols);
                    //                cout << "1, 3" << endl;
                }
                else if(max_dist == (v2-v4).norm())
                {
                    kl = MakeKeyLine(p2, p4, cur_img.cols);
                    //                cout << "1, 3" << endl;
                }
                else if(max_dist == (v1-v2).norm())
                {
                    kl = kl1;
                    //                cout << "1, 2" << endl;
                }
                else if(max_dist == (v3-v4).norm())
                {
                    kl = kl2;
                    //                cout << "3, 4" << endl;
                }

                double middist = sqrt(pow(kl1.pt.x - kl2.pt.x, 2) + pow(kl1.pt.y - kl2.pt.y, 2));

                if(abs(a-c) < 1 * M_PI/180 || abs(abs(a-c) - M_PI) < 1 * M_PI/180)
                {
                    if(abs(b-d) < 1 && middist < 5 && min_dist < 5)
                    {
                        kl.class_id = cur_keyLine[j].class_id;
                        cur_keyLine[j] = kl;
                        endflag = true;
                        break;
                    }
                    else if(kl1.size < 10)
                    {
                        endflag = true;
                        break;
                    }
                }
            }
            if (endflag)
            {
                endflag = false;
                continue;
            }
            kl1.class_id = cur_keyLine.size() + idx;
            cur_keyLine.push_back(kl1);
            ids.push_back(-1);
            track_cnt.push_back(1);

            idx++;
        }
    }

    Mat merged_img;
    Mat image = cur_img.clone();
    Mat canny = canny_img.clone();
    cvtColor(image, image, CV_GRAY2BGR);
    cvtColor(canny, canny, CV_GRAY2BGR);
    drawKeylines(image, cur_keyLine, image);
    hconcat(image, canny, merged_img);
    imshow("1111", merged_img);
    waitKey(1);

}

double LineFeatureTracker::Union_dist(VectorXd a, VectorXd b)
{
    double d = 0;
    if(a.size() == b.size())
        for(int i = 0; i < a.size(); i++)
            if(a(i) || b(i))
                d++;
    return d;
}

double LineFeatureTracker::Intersection_dist(VectorXd a, VectorXd b)
{
    double d = 0;
    if(a.size() == b.size())
        for(int i = 0; i < a.size(); i++)
            if(a(i) && b(i))
                d++;
    return d;
}

VectorXd LineFeatureTracker::Union(VectorXd a, VectorXd b)
{
    VectorXd d(a.size());
    if(a.size() == b.size())
        for(int i = 0; i < a.size(); i++)
            d(i) = (a(i) || b(i));

    return d;
}

VectorXd LineFeatureTracker::Intersection(VectorXd a, VectorXd b)
{
    VectorXd d(a.size());
    if(a.size() == b.size())
        for(int i = 0; i < a.size(); i++)
            d(i) = (a(i) && b(i));

    return d;
}

void LineFeatureTracker::getVPHypVia2Lines(vector<KeyLine> cur_keyLine, vector<Vector3d> &para_vector, vector<double> &length_vector, vector<double> &orientation_vector, std::vector<std::vector<Vector3d> > &vpHypo )
{
    int num = cur_keyLine.size();

    double noiseRatio = 0.5;
    double p = 1.0 / 3.0 * pow( 1.0 - noiseRatio, 2 );

    double confEfficience = 0.9999;
    int it = log( 1 - confEfficience ) / log( 1.0 - p );

    int numVp2 = 360;
    double stepVp2 = 2.0 * CV_PI / numVp2;

    ROS_DEBUG("get the parameters of each line");
    // get the parameters of each line
    for ( int i = 0; i < num; ++i )
    {
        Vector3d p1(cur_keyLine[i].getStartPoint().x, cur_keyLine[i].getStartPoint().y, 1.0);
        Vector3d p2(cur_keyLine[i].getEndPoint().x, cur_keyLine[i].getEndPoint().y, 1.0);

        para_vector.push_back(p1.cross( p2 ));

        double dx = cur_keyLine[i].getEndPoint().x - cur_keyLine[i].getStartPoint().x;
        double dy = cur_keyLine[i].getEndPoint().y - cur_keyLine[i].getStartPoint().y;
        length_vector.push_back(sqrt( dx * dx + dy * dy ));

        double orientation = atan2( dy, dx );
        if ( orientation < 0 )
        {
            orientation += CV_PI;
        }
        orientation_vector.push_back(orientation);
    }

    ROS_DEBUG("get vp hypothesis for each iteration");
    // get vp hypothesis for each iteration
    vpHypo = std::vector<std::vector<Vector3d> > ( it * numVp2, std::vector<Vector3d>(4) );
    int count = 0;
    srand((unsigned)time(NULL));
    for ( int i = 0; i < it; ++ i )
    {
//        ROS_DEBUG("i = %d", i);

        int idx1 = rand() % num;
        int idx2 = rand() % num;
        while ( idx2 == idx1 )
        {
            idx2 = rand() % num;
        }

        // get the vp1
        Vector3d vp1_Img = para_vector[idx1].cross( para_vector[idx2] );
        if ( vp1_Img(2) == 0 )
        {
            i --;
            continue;
        }

        Vector3d vp1(vp1_Img(0) / vp1_Img(2) - pinhole_camera->getParameters().cx(),
                     vp1_Img(1) / vp1_Img(2) - pinhole_camera->getParameters().cy(),
                     pinhole_camera->getParameters().fx() );
        if ( vp1(2) == 0 ) { vp1(2) = 0.0011; }
        double N = sqrt( vp1(0) * vp1(0) + vp1(1) * vp1(1) + vp1(2) * vp1(2) );
        vp1 *= 1.0 / N;

        // get the vp2 and vp3
        Vector3d vp2( 0.0, 0.0, 0.0 );
        Vector3d vp3( 0.0, 0.0, 0.0 );
        Vector3d vp4( 0.0, 0.0, 0.0 );

        for ( int j = 0; j < numVp2; ++ j )
        {
            // vp2
            double lambda = j * stepVp2;

            double k1 = vp1(0) * sin( lambda ) + vp1(1) * cos( lambda );
            double k2 = vp1(2);
            double phi = atan( - k2 / k1 );

            double Z = cos( phi );
            double X = sin( phi ) * sin( lambda );
            double Y = sin( phi ) * cos( lambda );

            vp2(0) = X;  vp2(1) = Y;  vp2(2) = Z;
            if ( vp2(2) == 0.0 ) { vp2(2) = 0.0011; }
            N = sqrt( vp2(0) * vp2(0) + vp2(1) * vp2(1) + vp2(2) * vp2(2) );
            vp2 *= 1.0 / N;
            if ( vp2(2) < 0 ) { vp2 *= -1.0; }

            // vp3
            vp3 = vp1.cross( vp2 );
            if ( vp3(2) == 0.0 ) { vp3(2) = 0.0011; }
            N = sqrt( vp3(0) * vp3(0) + vp3(1) * vp3(1) + vp3(2) * vp3(2) );
            vp3 *= 1.0 / N;
            if ( vp3(2) < 0 ) { vp3 *= -1.0; }
            //
            vpHypo[count][0] = Vector3d( vp1(0), vp1(1), vp1(2) );
            vpHypo[count][1] = Vector3d( vp2(0), vp2(1), vp2(2) );
            vpHypo[count][2] = Vector3d( vp3(0), vp3(1), vp3(2) );

            count ++;
        }
    }
    ROS_DEBUG("get vp hypothesis for each iteration, done");
}

void LineFeatureTracker::getSphereGrids(vector<KeyLine> cur_keyLine, vector<Vector3d> &para_vector, vector<double> &length_vector, vector<double> &orientation_vector, std::vector<std::vector<double> > &sphereGrid )
{
    // build sphere grid with 1 degree accuracy
    double angelAccuracy = 1.0 / 180.0 * CV_PI;
    double angleSpanLA = CV_PI / 2.0;
    double angleSpanLO = CV_PI * 2.0;
    int gridLA = angleSpanLA / angelAccuracy;
    int gridLO = angleSpanLO / angelAccuracy;

    sphereGrid = std::vector< std::vector<double> >( gridLA, std::vector<double>(gridLO) );
    for ( int i=0; i<gridLA; ++i )
    {
        for ( int j=0; j<gridLO; ++j )
        {
            sphereGrid[i][j] = 0.0;
        }
    }

    // put intersection points into the grid
    double angelTolerance = 60.0 / 180.0 * CV_PI;
    Vector3d ptIntersect;
    double x = 0.0, y = 0.0;
    double X = 0.0, Y = 0.0, Z = 0.0, N = 0.0;
    double latitude = 0.0, longitude = 0.0;
    int LA = 0, LO = 0;
    double angleDev = 0.0;
    for ( int i=0; i<cur_keyLine.size()-1; ++i )
    {
        for ( int j=i+1; j<cur_keyLine.size(); ++j )
        {
            ptIntersect = para_vector[i].cross( para_vector[j] );

            if ( ptIntersect(2) == 0 )
            {
                continue;
            }

            x = ptIntersect(0) / ptIntersect(2);
            y = ptIntersect(1) / ptIntersect(2);

            X = x - pinhole_camera->getParameters().cx();
            Y = y - pinhole_camera->getParameters().cy();
            Z = pinhole_camera->getParameters().fx();
            N = sqrt( X * X + Y * Y + Z * Z );

            latitude = acos( Z / N );
            longitude = atan2( X, Y ) + CV_PI;

            LA = int( latitude / angelAccuracy );
            if ( LA >= gridLA )
            {
                LA = gridLA - 1;
            }

            LO = int( longitude / angelAccuracy );
            if ( LO >= gridLO )
            {
                LO = gridLO - 1;
            }

            //
            angleDev = abs( orientation_vector[i] - orientation_vector[j] );
            angleDev = min( CV_PI - angleDev, angleDev );
            if ( angleDev > angelTolerance )
            {
                continue;
            }

            sphereGrid[LA][LO] += sqrt( length_vector[i] * length_vector[j] ) * ( sin( 2.0 * angleDev ) + 0.2 ); // 0.2 is much robuster
        }
    }

    //
    int halfSize = 1;
    int winSize = halfSize * 2 + 1;
    int neighNum = winSize * winSize;

    // get the weighted line length of each grid
    std::vector< std::vector<double> > sphereGridNew = std::vector< std::vector<double> >( gridLA, std::vector<double>(gridLO) );
    for ( int i=halfSize; i<gridLA-halfSize; ++i )
    {
        for ( int j=halfSize; j<gridLO-halfSize; ++j )
        {
            double neighborTotal = 0.0;
            for ( int m=0; m<winSize; ++m )
            {
                for ( int n=0; n<winSize; ++n )
                {
                    neighborTotal += sphereGrid[i-halfSize+m][j-halfSize+n];
                }
            }

            sphereGridNew[i][j] = sphereGrid[i][j] + neighborTotal / neighNum;
        }
    }
    sphereGrid = sphereGridNew;
}

void LineFeatureTracker::getBestVpsHyp( std::vector<std::vector<double> > &sphereGrid, std::vector<std::vector<Vector3d> >  &vpHypo, std::vector<Vector3d> &vps  )
{
    int num = vpHypo.size();
    double oneDegree = 1.0 / 180.0 * CV_PI;

    // get the corresponding line length of every hypotheses
    std::vector<double> lineLength( num, 0.0 );
    for ( int i = 0; i < num; ++ i )
    {
        std::vector<cv::Point2d> vpLALO( 3 );
        for ( int j = 0; j < 3; ++ j )
        {
            if ( vpHypo[i][j](2) == 0.0 )
            {
                continue;
            }

            if ( vpHypo[i][j](2) > 1.0 || vpHypo[i][j](2) < -1.0 )
            {
                cout<<1.0000<<endl;
            }
            double latitude = acos( vpHypo[i][j](2) );
            double longitude = atan2( vpHypo[i][j](0), vpHypo[i][j](1) ) + CV_PI;

            int gridLA = int( latitude / oneDegree );
            if ( gridLA == 90 )
            {
                gridLA = 89;
            }

            int gridLO = int( longitude / oneDegree );
            if ( gridLO == 360 )
            {
                gridLO = 359;
            }

            lineLength[i] += sphereGrid[gridLA][gridLO];
        }
    }

    // get the best hypotheses
    int bestIdx = 0;
    double maxLength = 0.0;
    for ( int i = 0; i < num; ++ i )
    {
        if ( lineLength[i] > maxLength )
        {
            maxLength = lineLength[i];
            bestIdx = i;
        }
    }

    vps = vpHypo[bestIdx];
//    cout << vps.size() << endl;
}

void LineFeatureTracker::lines2Vps(vector<KeyLine> cur_keyLine, double thAngle, std::vector<Vector3d> &vps, std::vector<std::vector<int> > &clusters, vector<int> &vp_idx)
{
    clusters.clear();
    clusters.resize( 3 );

    int vps_size = 3;
    //get the corresponding vanish points on the image plane
    std::vector<cv::Point2d> vp2D( vps_size );
    for ( int i = 0; i < vps_size; ++ i )
    {
        vp2D[i].x =  vps[i](0) * pinhole_camera->getParameters().fx() /
                     vps[i](2) + pinhole_camera->getParameters().cx();
        vp2D[i].y =  vps[i](1) * pinhole_camera->getParameters().fy() /
                     vps[i](2) + pinhole_camera->getParameters().cy();
    }

    for ( int i = 0; i < cur_keyLine.size(); ++ i )
    {
        double x1 = cur_keyLine[i].getStartPoint().x;
        double y1 = cur_keyLine[i].getStartPoint().y;
        double x2 = cur_keyLine[i].getEndPoint().x;
        double y2 = cur_keyLine[i].getEndPoint().y;
        double xm = ( x1 + x2 ) / 2.0;
        double ym = ( y1 + y2 ) / 2.0;

        double v1x = x1 - x2;
        double v1y = y1 - y2;
        double N1 = sqrt( v1x * v1x + v1y * v1y );
        v1x /= N1;   v1y /= N1;

        double minAngle = 1000.0;
        int bestIdx = 0;
        for ( int j = 0; j < vps_size; ++ j )
        {
            double v2x = vp2D[j].x - xm;
            double v2y = vp2D[j].y - ym;
            double N2 = sqrt( v2x * v2x + v2y * v2y );
            v2x /= N2;  v2y /= N2;

            double crossValue = v1x * v2x + v1y * v2y;
            if ( crossValue > 1.0 )
            {
                crossValue = 1.0;
            }
            if ( crossValue < -1.0 )
            {
                crossValue = -1.0;
            }
            double angle = acos( crossValue );
            angle = min( CV_PI - angle, angle );

            if ( angle < minAngle )
            {
                minAngle = angle;
                bestIdx = j;
            }
        }

        //
        if ( minAngle < thAngle )
        {
            clusters[bestIdx].push_back( i );
            vp_idx.push_back(bestIdx);
        }
        else
            vp_idx.push_back(3);
    }
}

void LineFeatureTracker::drawClusters( cv::Mat &img, std::vector<KeyLine> &lines, std::vector<std::vector<int> > &clusters )
{
    Mat vp_img = img.clone();
    Mat line_img = img.clone();
    int cols = img.cols;
    int rows = img.rows;

    //draw lines
    std::vector<cv::Scalar> lineColors( 3 );
    lineColors[0] = cv::Scalar( 0, 0, 255 );
    lineColors[1] = cv::Scalar( 0, 255, 0 );
    lineColors[2] = cv::Scalar( 255, 0, 0 );
//    lineColors[3] = cv::Scalar( 0, 255, 255 );

    for ( int i=0; i<lines.size(); ++i )
    {
        int idx = i;
        cv::Point2f pt_s = lines[i].getStartPoint();
        cv::Point2f pt_e = lines[i].getEndPoint();
        cv::Point pt_m = ( pt_s + pt_e ) * 0.5;

        cv::line( vp_img, pt_s, pt_e, cv::Scalar(0,0,0), 2, CV_AA );
        cv::line( line_img, pt_s, pt_e, cv::Scalar(0,255,255), 2, CV_AA );
    }

    for ( int i = 0; i < clusters.size(); ++i )
    {
        for ( int j = 0; j < clusters[i].size(); ++j )
        {
            int idx = clusters[i][j];

            cv::Point2f pt_s = lines[idx].getStartPoint();
            cv::Point2f pt_e = lines[idx].getEndPoint();
            cv::Point pt_m = ( pt_s + pt_e ) * 0.5;

            cv::line( vp_img, pt_s, pt_e, lineColors[i], 2, CV_AA );
        }
    }
    imshow("img", img);
    imshow("line img", line_img);
    imshow("vp img", vp_img);
    waitKey(1);
}

double LineFeatureTracker::SafeAcos (double x)
{
    if (x < -1.0) x = -1.0 ;
    else if (x > 1.0) x = 1.0 ;
    return acos(x) ;
}

void LineFeatureTracker::removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();
    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);
    matrix.conservativeResize(numRows,numCols);
}

void LineFeatureTracker::removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;
    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);
    matrix.conservativeResize(numRows,numCols);
}

void LineFeatureTracker::getUndistortEndpointsXY(const vector<Line>& lines, vector<Point2f> &endpoints) {
    endpoints.clear();
    int lines_size = lines.size();
    endpoints.resize(2 * lines_size);
    for (int i = 0; i < lines_size; ++i)
    {
        endpoints[2 * i] = lines[i].start_xy;
        endpoints[2 * i + 1] = lines[i].end_xy;
    }
}

void LineFeatureTracker::getUndistortSample(const vector<Line>& lines, vector<cv::Point2f> &points, vector<int> &point2lineIdx) {
    points.clear();
    point2lineIdx.clear();
    int lines_size = lines.size();
    for (int i = 0; i < lines_size; ++i)
        for (int j = 0; j < lines[i].sample_points_undistort.size(); ++j) {
            points.emplace_back(lines[i].sample_points_undistort[j]);
            point2lineIdx.emplace_back(i); //record local line index
        }
//    for (int i = 0; i < point2lineIdx.size(); ++i)
//        ROS_DEBUG("point %d  --->  line local id %d", i, point2lineIdx[i]);

}

void LineFeatureTracker::initLinesPredict() {
    lines_predict = curr_line;
    for (Line& line : lines_predict)
    {
        line.start_predict_fail = false;
        line.end_predict_fail = false;
        line.updated_forwframe = false; //非新检测
        line.extended = false;
        line.sample_points_undistort.clear();
    }
    prev_lineID = ids;
    tmp_track_cnt = track_cnt;
//    ROS_DEBUG("lines_predict: %d, prev_lineID: %d, tmp_track_cnt: %d", lines_predict.size(), prev_lineID.size(), tmp_track_cnt.size());
//
//    ROS_ASSERT(lines_predict.size() == prev_lineID.size());

    ROS_WARN("lines_predict");
    for (int i = 0; i < lines_predict.size(); ++i)
    {
        const Line& l = lines_predict[i];
        ROS_DEBUG("line local: %d , global: %d", i, prev_lineID[i]);
    }
}

void LineFeatureTracker::rejectWithF(vector<uchar> &status) {
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        status.clear();
        TicToc t_f;
        cv::Mat matrix_f =  cv::findFundamentalMat(cur_pts, forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void LineFeatureTracker::markFailEndpoints(const vector<uchar> &status) {
    for (int i = 0; i < int(forw_pts.size()) && i < curr_line.size() * 2; i++)
        if (!status[i] || !inBorder(forw_pts[i], forw_img.rows, forw_img.cols))
        {
//                ROS_WARN("status[%d] (%f, %f), (%f, %f)", i, forw_pts_tmp[i].x, forw_pts_tmp[i].y, forw_pts[i].x,forw_pts[i].y);
            if (i % 2 == 0) {
                lines_predict[i / 2].start_predict_fail = true;
//                ROS_DEBUG("line %d start point (%f, %f) predict fail", i / 2, lines_predict[i/2].start_xy.x,lines_predict[i/2].start_xy.y);
            }
            else {
                lines_predict[i / 2].end_predict_fail = true;
//                ROS_DEBUG("line %d end point (%f, %f)predict fail", i / 2, lines_predict[i/2].start_xy.x,lines_predict[i/2].start_xy.y);
            }
            lines_predict[i / 2].is_valid = Line::bad_prediction;
//            lines_predict[i / 2].is_valid = Line::endpoint_out_of_image;
//            ROS_DEBUG("line %d predict fail", i / 2);
        }
}

void LineFeatureTracker::markFailSamplePoints(const vector<uchar> &status, vector<int> &point2lineIdx) {
    const int num_line_endpoints = lines_predict.size() * 2;
    for (int i = 0; i < (int)point2lineIdx.size(); i++) {

        int point_id = i + num_line_endpoints;
        if (!status[point_id] || !inBorder(forw_pts[point_id], forw_img.rows, forw_img.cols))
            continue;
        const int& local_id = point2lineIdx[i];
//        ROS_DEBUG("forw_pts %d (%f, %f) ---> line local id: %d", i, forw_pts[point_id].x, forw_pts[point_id].y, local_id);
        lines_predict[local_id].sample_points_undistort.emplace_back(forw_pts[point_id]);
    }
}

void LineFeatureTracker::checkEndpointsAndUpdateLinesInfo() {
    //更新有效光流端点、直线信息
    ROS_DEBUG("check Optical Flow Endpoints");
    checkOpticalFlowEndpoints(lines_predict, forw_pts);

    //处理顶点光流跟丟的直线
    TicToc t_update;
    ROS_DEBUG("check And Update Endpoints");
    checkAndUpdateEndpoints(lines_predict);
    ROS_DEBUG("check And Update Endpoints: %fms", t_update.toc());
}

void LineFeatureTracker::checkOpticalFlowEndpoints(vector<Line> &lines, const vector<Point2f> &endpoints) {
    ROS_ASSERT(lines.size() * 2 <= endpoints.size());

    for (int i = 0; i < lines.size(); ++i)
    {
        const Point2f& start = endpoints[2 * i];
        const Point2f& end = endpoints[2 * i + 1];
        //不一定准确，因为光流预测失败不能说明端点真的在图像外
        if ((start.x < 0 && end.x < 0) || (start.x >= curr_img.cols && end.x >= curr_img.cols) ||
            (start.y < 0 && end.y < 0) || (start.y >= curr_img.rows && end.y >= curr_img.rows))
        {
            lines[i].is_valid = Line::line_out_of_view;
//            ROS_DEBUG("line[%d} out_of_view", i);
            continue;
        }

        if (!lines[i].start_predict_fail && !lines[i].end_predict_fail)
        {
            //延伸顶点
//            Point2f start_new, end_new;
//            if (extendEndpoints(forwframe_->img, start, end, start_new, end_new))
//                lines[i].extended = true;
//            MakeALine(start_new, end_new, lines[i]); //update lines information
            MakeALine(start, end, lines[i]); //update lines information
//            ROS_DEBUG("line %d, start (%f, %f), end (%f, %f)", i, start.x, start.y, end.x, end.y);
            lines[i].updated_forwframe = false;
        }
    }
}

void LineFeatureTracker::MakeALine(cv::Point2f start_pts, cv::Point2f end_pts, const int &rows, const int &cols,
                                   Line &line) {
    // Set start point(and octave)
    if(start_pts.x > end_pts.x)
        swap(start_pts, end_pts);

    // correct endpoints
    if (start_pts.x < 0 || start_pts.y < 0)
    {
//        ROS_DEBUG("correctNegativeXY: (%f. %f)", start_pts.x, start_pts.y);
        correctNegativeXY(start_pts, end_pts);
//        ROS_DEBUG("after correctNegativeXY: (%f. %f)", start_pts.x, start_pts.y);
    }

    if (end_pts.x >= cols || end_pts.y >= rows)
    {
//        ROS_DEBUG("correctOutsideXY end: (%f. %f)", end_pts.x, end_pts.y);
        correctOutsideXY(start_pts, end_pts, rows, cols);
//        ROS_DEBUG("after correctOutsideXY end: (%f. %f)", end_pts.x, end_pts.y);
    }

    if (start_pts.x >= cols || start_pts.y >= rows)
    {
//        ROS_DEBUG("correctOutsideXY start: (%f. %f)", start_pts.x, start_pts.y);
        correctOutsideXY(end_pts, start_pts, rows, cols);
//        ROS_DEBUG("after correctOutsideXY start: (%f. %f)", start_pts.x, start_pts.y);
    }

    line.start_xy = start_pts;
//    keyLine.sPointInOctaveX = start_pts.x;
//    keyLine.sPointInOctaveY = start_pts.y;

    // Set end point(and octave)
    line.end_xy = end_pts;
//    keyLine.ePointInOctaveX = end_pts.x;
//    keyLine.ePointInOctaveY = end_pts.y;

    line.StartPt = line.start_xy;
    line.EndPt = line.end_xy;

    // Set angle
    line.theta = atan2((end_pts.y-start_pts.y),(end_pts.x-start_pts.x));
//    ROS_DEBUG(" line.theta  = %f",  line.theta / M_PI * 180);

    // Set line length & response
//    keyLine.lineLength = keyLine.numOfPixels = norm( Mat(end_pts), Mat(start_pts));
    line.length = norm( Mat(end_pts), Mat(start_pts));

//    keyLine.response = norm( Mat(end_pts), Mat(start_pts))/cols;

    // Set octave
//    keyLine.octave = 0;

    // Set pt(mid point)
    line.Center = (start_pts + end_pts)/2;

    // Set size
    line.unitDir = (end_pts - start_pts) / line.length;

//    line.new_detect = true;
}

void LineFeatureTracker::MakeALine(cv::Point2f start_pts, cv::Point2f end_pts, Line &line) {
//    // Set start point(and octave)
    if(start_pts.x > end_pts.x)
        swap(start_pts, end_pts);

    line.start_xy = start_pts;

//    keyLine.sPointInOctaveX = start_pts.x;
//    keyLine.sPointInOctaveY = start_pts.y;

    // Set end point(and octave)
    line.end_xy = end_pts;
//    keyLine.ePointInOctaveX = end_pts.x;
//    keyLine.ePointInOctaveY = end_pts.y;

    line.StartPt = line.start_xy;
    line.EndPt = line.end_xy;

    // Set angle
    line.theta = atan2((end_pts.y-start_pts.y),(end_pts.x-start_pts.x));

    // Set line length & response
//    keyLine.lineLength = keyLine.numOfPixels = norm( Mat(end_pts), Mat(start_pts));
    line.length = norm( Mat(end_pts), Mat(start_pts));

//    keyLine.response = norm( Mat(end_pts), Mat(start_pts))/cols;

    // Set octave
//    keyLine.octave = 0;

    // Set pt(mid point)
    line.Center = (start_pts + end_pts)/2;

    // Set size
    line.unitDir = (end_pts - start_pts) / line.length;
}

void LineFeatureTracker::checkAndUpdateEndpoints(vector<Line> &lines) {

    const float threshold = 0.85;
    int i = 0;
    for (Line& line : lines)
    {
        //todo 采样点和端点数量
        if (line.sample_points_undistort.size() < 3) {
            line.is_valid = Line::bad_prediction;
            continue;
        }

//        ROS_DEBUG("line[%d]", i);

        bool need_update = false;
        //check start point
//        if (line.start_predict_fail)
        if (line.start_predict_fail || !inBorder(line.start_xy, forw_img.rows, forw_img.cols))
        {
            //todo
            if (!line.sample_points_undistort.empty()) {
                line.start_xy = line.sample_points_undistort.front();
                need_update = true;
            }


////            for (int i = 1; i < line.sample_zncc.size(); ++i)
//            if (!line.sample_points_undistort.empty())
//                line.popFront();
//            while (!line.sample_points_undistort.empty())
//            {
////                ROS_DEBUG("line.sample_zncc.front(): %f", line.sample_zncc.front());
//                if (line.sample_zncc.front() < threshold)
//                    line.popFront();
//                else
//                {
//                    //                line.StartUV = line.sample_points_undistort[i];
//                    line.StartUV = line.sample_points_undistort.front();
//                    line.start_predict_fail = false;
//                    break;
//                }
//            }
        }

        //check end point
//        if (line.end_predict_fail)
        if (line.end_predict_fail || !inBorder(line.end_xy, forw_img.rows, forw_img.cols))
        {
            if (!line.sample_points_undistort.empty()) {
                line.end_xy = line.sample_points_undistort.back();
                need_update = true;
            }

////            for (int i = line.sample_zncc.size() - 1; i >= 0; --i)
////            ROS_DEBUG("line.sample_zncc.back(): %f", line.sample_zncc.back());
//            if (!line.sample_points_undistort.empty())
//                line.popBack();
//
//            while (!line.sample_points_undistort.empty())
//            {
//                if (line.sample_zncc.back() < threshold)
//                    line.popBack();
//                else
//                {
//                    //line.EndUV = line.sample_points_undistort[i];
//                    line.EndUV = line.sample_points_undistort.back();
//                    line.end_predict_fail = false;
//                    break;
//                }
//            }
        }

        if (need_update)
        {
//            MakeALine(line.start_xy, line.end_xy, line);
            line.is_valid = Line::valid;
            line.updated_forwframe = false;
//            ROS_DEBUG("line[%d](new_detect = %d) StartUV(%f, %f)", i, line.new_detect, line.StartUV.x, line.StartUV.y);
//            ROS_DEBUG("line[%d](new_detect = %d) EndUV(%f, %f)", i,  line.new_detect, line.EndUV.x, line.EndUV.y);
        }
//        //check new line
        if (line.length < 30)
            line.is_valid = Line::too_short;
        ++i;
    }
}

void LineFeatureTracker::reduceLine(vector<Line> &lines, vector<int> &IDs) {
    ROS_ASSERT(lines.size() == IDs.size());
    ROS_ASSERT(lines.size() == tmp_track_cnt.size());
    ROS_WARN("reduceLine");

    int j = 0;
    for (int i = 0; i < int(lines.size()); i++)
    {
        if (lines[i].is_valid == Line::valid)
        {
            lines[j] = lines[i];
            tmp_track_cnt[j] = tmp_track_cnt[i];
            forw_pts[2 * j] = forw_pts[2 * i];
            forw_pts[2 * j + 1] = forw_pts[2 * i + 1];
            IDs[j++] = IDs[i];
        }
        else
            ROS_WARN("lose line local: %d, global: %d, valid type = %d", i, IDs[i], lines[i].is_valid);
    }

    forw_pts.resize(2 * j);
    tmp_track_cnt.resize(j);
    lines.resize(j);
    IDs.resize(j);
}

void LineFeatureTracker::fitLinesBySamples(vector<Line> &lines) {
    cv::Vec4f line4f;
    for (Line& line : lines) {
        std::vector<Point2f> points(line.sample_points_undistort.begin(), line.sample_points_undistort.end());
        cv::fitLine(points, line4f, CV_DIST_HUBER, 1.0, 0.1, 0.01);
        Point2f dir(line4f[0], line4f[1]);
        Point2f x0y0(line4f[2], line4f[3]);
        Point2f line_start = x0y0 + (line.start_xy - x0y0).dot(dir) * dir;
        Point2f line_end = x0y0 + (line.end_xy - x0y0).dot(dir) * dir;
        MakeALine(line_start, line_end, forw_img.rows, forw_img.cols, line);
    }
}

void LineFeatureTracker::extractELSEDLine(const Mat &img, vector<Line> &lines, vector<Line> &lines_exist) {
    upm::ELSED elsed;
    elsed.setNMSExtend(nms_extend);

    lines.clear();
    TicToc t_r;

    //todo 采样点传递到ELSED，验证是否为anchor
    if (!lines_exist.empty())
    {
        int line_size = lines_exist.size();
        elsed.getLinesExist().resize(line_size);
        int i = 0;
        for (const Line& line : lines_exist)
        {
            upm::lineInfo& line_info = elsed.getLinesExist()[i];
            line_info.centerPoint = line.Center;
            line_info.angle =  line.theta / M_PI * 180.0;
            line_info.length = line.length;
            line_info.startPoint = line.start_xy;
            line_info.endPoint = line.end_xy;
            line_info.unitDir = line.unitDir;
//            line_info.need_detect = !line.extended;
            line_info.need_detect = true;
            std::vector<cv::Point2f> samples(line.sample_points_undistort.begin(), line.sample_points_undistort.end());
            swap(samples, line_info.sample_points);
            ++i;
        }
    }

    if (lines_exist.size() >= max_lines_num)
        elsed.enough_lines = true;

    //    upm::Segments segs = elsed.detect(img);
    ROS_DEBUG("process Image");
    elsed.processImage(img);
    ROS_DEBUG("process Image, done");

//    computeGradientDirection(elsed.getImgInfoPtr()->dxImg, elsed.getImgInfoPtr()->dyImg, gradient_dir);


//    segs = elsed.getELSEDSegments();
    int line_id = 0;
    const upm::Segments& segs = elsed.outside_nms_lines;
    lines.resize(segs.size());
//    ROS_DEBUG("new lines: %d", segs.size());
    for(unsigned int i = 0; i < segs.size(); i++)
    {
        Line& line = lines[i];
        MakeALine(cv::Point2f(segs[i][0], segs[i][1]), cv::Point2f(segs[i][2], segs[i][3]), img.rows, img.cols, line);
        line.start_xy_visual = line.start_xy;
        line.end_xy_visual = line.end_xy;
        line.updated_forwframe = true;
        line.id = line_id;
        line.num_untracked = 0;
        line_id++;
//        ROS_DEBUG("line[%d] length: %f", i, line.length);
    }

    std::vector<upm::Segments> nms_lines;
    nms_lines = elsed.getNMSSegments();
    std::vector<std::vector<Line>> lines_predict_extract;
    lines_predict_extract.resize(nms_lines.size());
//    line_update.clear();
    for(unsigned int i = 0; i < nms_lines.size(); i++)
    {
        lines_predict_extract[i].resize(nms_lines[i].size());
        for (int j = 0; j < nms_lines[i].size(); ++j) {
            Line& line = lines_predict_extract[i][j];
            MakeALine(cv::Point2f(nms_lines[i][j][0], nms_lines[i][j][1]), cv::Point2f(nms_lines[i][j][2], nms_lines[i][j][3]), img.rows, img.cols, line);
            line.updated_forwframe = true;
            line.id = line_id;
            line_id++;
//            line_update.emplace_back(line);
        }
//        ROS_DEBUG("nms[%d] lines: %d", i, nms_lines[i].size());
    }

    //check lines similarity
    ROS_DEBUG("update lines_exist");
    for (int i = 0; i < lines_exist.size(); ++i)
    {
//        ROS_DEBUG("i = %d", i);
        Line& line_old = lines_exist[i];
//        line_old.Center;
//        float angle_old = line_old.theta / M_PI * 180.0;
//        line_old.length;
//        line_old.StartUV;
//        line_old.EndUV;
        const float mid_distance_threshold = 8.0;

        int id_min = -1;
        float dist_min = 9999.999;
        bool merge_line = false;//todo
        vector<int> id_merge;
//        ROS_DEBUG("lines_predict_extract[%d].size() = %lu", i, lines_predict_extract[i].size());
        ROS_ASSERT(lines_predict_extract.size() == lines_exist.size());
        //Traverse all lines in the NMS region to update the line
        for (int j = 0; j < lines_predict_extract[i].size(); ++j)
        {
//            ROS_DEBUG("j = %d", j);
            const Line& line_new = lines_predict_extract[i][j];

            //same line
            if (norm((line_old.start_xy - line_new.start_xy)) < mid_distance_threshold &&
                norm((line_old.end_xy - line_new.end_xy)) < mid_distance_threshold)
            {
//                ROS_DEBUG("line %d, find same line", j);
                line_old = line_new;
                line_old.updated_forwframe = true;
                line_old.start_xy_visual = line_old.start_xy;
                line_old.end_xy_visual = line_old.end_xy;
                line_old.num_untracked = 0;
                id_merge.clear();
                break;
            }

            //similar length
            if (line_old.length >= line_new.length) {
                if (line_old.length * 0.5 > line_new.length)
                    continue;
            } else {
                if (line_new.length * 0.5 > line_old.length)
                    continue;
            }

            //todo angle
            if (acos(abs(line_old.unitDir.dot(line_new.unitDir))) > 8.0 / 180.0 * M_PI)
                continue;
            //distance

            //check midpoint distance
            if (!checkMidPointDistance(line_old.start_xy, line_old.end_xy, line_new.start_xy, line_new.end_xy, mid_distance_threshold))
                continue;

            float d1 = point2line(line_old.start_xy, line_new.start_xy, line_new.end_xy);
            float d2 = point2line(line_old.end_xy, line_new.start_xy, line_new.end_xy);
//            float d3 = point2line(line_old.Center, line_new.start_xy, line_new.end_xy);
//            float d_mean = (d1 + d2 + d3) / 3.0;
            float d_mean = (d1 + d2) / 2.0;
            if (merge_line) {
                if (d_mean < 5)
                    id_merge.emplace_back(j);
            } else {
                if (dist_min > d_mean)
                    dist_min = d_mean;
                id_min = j;
            }
        }
//        ROS_DEBUG("compute similarity, done");

        if ((merge_line && id_merge.empty()) || (!merge_line && id_min == -1)) {
            line_old.num_untracked++;
            if (line_old.num_untracked >= LINE_MAX_UNTRACKED)
                line_old.is_valid = Line::large_untracked;
            line_old.updated_forwframe = false;
            continue;
        }

        if (!merge_line)
            line_old = lines_predict_extract[i][id_min];
        else
//        if (!id_merge.empty())
        {
            Line line = lines_predict_extract[i][id_merge[0]];

            for (int j = 1; j < id_merge.size(); ++j)
            {
                const Line& line_j = lines_predict_extract[i][id_merge[j]];

                cv::Point2f start_new, end_new;
                if (line.start_xy.x < line_j.start_xy.x)
                {
                    start_new = line.start_xy;
//                    start_is_old = true;
                }
                else
                {
                    start_new = line_j.start_xy;
//                    start_is_old = false;
                }

                if (line.end_xy.x > line_j.end_xy.x)
                {
                    end_new = line.end_xy;
//                    end_is_old = true;
                }
                else
                {
                    end_new = line_j.end_xy;
//                    end_is_old = false;
                }

                MakeALine(start_new, end_new, img.rows, img.cols, line);
            }
            line_old = line;
        }
        line_old.updated_forwframe = true;
        line_old.start_xy_visual = line_old.start_xy;
        line_old.end_xy_visual = line_old.end_xy;
        line_old.num_untracked = 0;
//        ROS_DEBUG("merge, done");
    }
    ROS_DEBUG("update lines_exist, done");

    //排序后id会不对应
//    sort(lines_exist.begin(),lines_exist.end(), Line::sortByLength);
//    ROS_DEBUG("update lines_exist, done");

//    ROS_WARN("ELSED lines_exist");
//    for (int i = 0; i < lines_exist.size(); ++i)
//    {
//        const Line& l = lines_exist[i];
//        ROS_DEBUG("line id: %d , new_detect : %d", prev_lineID[i], (int)l.new_detect);
//        ROS_DEBUG("start visual: (%f, %f)", l.start_xy_visual.x, l.start_xy_visual.y);
//        ROS_DEBUG("end visual:   (%f, %f)", l.end_xy_visual.x,  l.end_xy_visual.y);
//    }
}

bool LineFeatureTracker::checkMidPointDistance(const Point2f &start_i, const Point2f &end_i, const Point2f &start_j,
                                               const Point2f &end_j, const float threshold) {
    const Point2f mid_i(start_i * 0.5 + end_i * 0.5);
    const Point2f mid_j(start_j * 0.5 + end_j * 0.5);
    float d1 = point2line(mid_i, start_j, end_j);
    float d2 = point2line(mid_j, start_i, end_i);
//    ROS_DEBUG("d1 = %f, d2 = %f", d1, d2);
    if(d1 <= threshold && d2 <= threshold)
        return true;
    return false;
}

void LineFeatureTracker::updateTrackingLinesAndID() {
    ROS_ASSERT(lines_predict.size() == prev_lineID.size());
    ROS_ASSERT(lines_predict.size() == tmp_track_cnt.size());

    for (int i = 0; i < lines_predict.size(); ++i)
        tmp_track_cnt[i]++;

    int num_lines_predict = lines_predict.size();
    int new_lines_needed = max_lines_num - num_lines_predict;
    int num_lines_new = forw_line.size();
    if (new_lines_needed > 0)
    {
//        sort(forwframe_->mergedLine.begin(),forwframe_->mergedLine.end(), Line::sortByLength);
        // set new lines ID
        for (size_t i = 0; i < new_lines_needed && i < num_lines_new; ++i)
        {
            lines_predict.emplace_back(forw_line[i]);
            prev_lineID.emplace_back(allfeature_cnt++);
            tmp_track_cnt.emplace_back(1);
        }
    }
//    swap(lines_predict, forwframe_->mergedLine);
//    swap(prev_lineID, forwframe_->lineID);

    forw_line = lines_predict;
    ids = prev_lineID;
    track_cnt = tmp_track_cnt;
//    for (const int& id : forwframe_->lineID)
//        ROS_INFO("line id: %d", id);
}

void LineFeatureTracker::checkGradient(vector<Line> &lines, const int &start_idx) {
    for (unsigned int i = start_idx; i <lines.size(); i++)
    {
        Line& line = lines[i];

        line.sample_points_undistort.clear();

        float line_angle = line.theta / M_PI * 180.0;
        if (line_angle < 0)
            line_angle += 180.0;
        float angle_threshold = 30.0; //angle between line and sample points gradient direction

//        //鱼眼重投影 无畸变 像素坐标
        Eigen::Vector2f undistort_start, undistort_end;
        undistort_start(0) = line.start_xy.x;
        undistort_start(1) = line.start_xy.y;
        undistort_end(0) = line.end_xy.x;
        undistort_end(1) = line.end_xy.y;
        const float &length = line.length;

//        //sample points from start to end
        deque<Point2f>& sample_pts = line.sample_points_undistort;

        //todo 按像素距离、间隔较小值采样 调参
        Eigen::Vector2f start_end = undistort_end - undistort_start;
        double times = max(10.0,  start_end.norm() / 10.0);
        Eigen::Vector2f interval = (undistort_end - undistort_start) / times;
        Eigen::Vector2f undistort_p = undistort_start + interval;
        int pts_total = 0, pts_good = 0;
        for (int step = 0; step < times - 1; ++step, undistort_p += interval)
        {
            ++pts_total;
//            if (step == times) //end point
//            {
//                undistort_p(0) =  undistort_end(0);
//                undistort_p(1) =  undistort_end(1);
//            }

            //非端点加入梯度方向条件，不满足不采样
//            if (step != 0 || step != times) // not start point
            {
                if (largeGradientAngle(line_angle, undistort_p(0), undistort_p(1), angle_threshold))
                    continue;
            }

            ++pts_good;
            sample_pts.push_back(Point2f(undistort_p(0), undistort_p(1)));
        }

        //todo
        if ((pts_good * 1.0) / (pts_total * 1.0) <= 0.5)
        {
            line.is_valid = Line::bad_gradient_direction;
        }
    }
}

bool LineFeatureTracker::largeGradientAngle(const float &line_angle, const float &x, const float &y,
                                            const float &threshold) {
    ROS_ASSERT(dxImg.ptr<int16_t>());
    ROS_ASSERT(dyImg.ptr<int16_t>());

    const int16_t& dx = dxImg.at<int16_t>(y, x);
    const int16_t& dy = dyImg.at<int16_t>(y, x);
    if (sqrt(dx * dx + dy *dy) < 30) //gradient too small
        return true;
    float gradient_angle = atan2(dy, dx) / M_PI * 180.0; //degree
//                float gradient_angle = gradient_dir.at<float>(undistort_p(1), undistort_p(0));//degree
    if (gradient_angle < 0.0)
        gradient_angle += 180.0;
    else if (gradient_angle > 180.0)
        gradient_angle -= 180.0;
    float delta_angle = abs(gradient_angle - line_angle);
    if (delta_angle < 90.0 - threshold ||  delta_angle > 90 + threshold)
        return true;
    return false;}

void LineFeatureTracker::Line2KeyLine(const vector<Line> &lines_in, vector<LineKL> &lines_out) {
    int num_lines = lines_in.size();
    lines_out.resize(num_lines);
    int line_id = 0;
    for (int i = 0; i < num_lines; ++i) {
        const Line& line = lines_in[i];
        lines_out[i] = MakeKeyLine(line.start_xy, line.end_xy, forw_img.cols);
//        ROS_DEBUG("lines_out[%d]: start(%f, %f) end(%f, %f)", i, lines_out[i].startPointX,lines_out[i].startPointY
//                  , lines_out[i].endPointX, lines_out[i].endPointY);
//        lines_out[i].class_id = line_id;
//        ++line_id;
    }
}

void LineFeatureTracker::KeyLine2Line(const vector<LineKL> &lines_in, vector<Line> &lines_out) {
    int num_lines = lines_in.size();
    lines_out.resize(num_lines);
    for (int i = 0; i < num_lines; ++i) {
        const KeyLine& line = lines_in[i];
        MakeALine(line.getStartPoint(), line.getEndPoint(), lines_out[i]);
    }
}

void LineFeatureTracker::correctNegativeXY(Point2f &start_pts, Point2f &end_pts) {
    cv::Point2f dir = end_pts - start_pts;
    float length = norm(dir);
    dir = dir / length;
//    ROS_DEBUG("dir: (%f. %f)", dir.x, dir.y);
    if (start_pts.x < 0)
    {
        float delta_x = 0 - start_pts.x;
        start_pts.x = 0;
        if (dir.x != 0)
            start_pts.y = start_pts.y + delta_x / dir.x * dir.y;
    }
    if (start_pts.y < 0)
    {
        float delta_y = 0 - start_pts.y;
        start_pts.y = 0;
        if (dir.y != 0)
            start_pts.x = start_pts.x + delta_y / dir.y * dir.x;
    }
}

void LineFeatureTracker::correctOutsideXY(Point2f &start_pts, Point2f &end_pts, const int &rows, const int &cols) {
    cv::Point2f dir = end_pts - start_pts;
    float length = norm(dir);
    dir = dir / length;
//    ROS_DEBUG("dir after: (%f, %f)", dir.x, dir.y);

    if (end_pts.x >= cols)
    {
        float delta_x = cols - 1 - end_pts.x;
        end_pts.x = cols - 1;
        if (dir.x != 0)
            end_pts.y = end_pts.y + delta_x / dir.x * dir.y;
//        ROS_WARN("end_pts X: (%f. %f)", end_pts.x, end_pts.y);

    }
//    ROS_WARN("end_pts X: (%f. %f)", end_pts.x, end_pts.y);

    if (end_pts.y >= rows)
    {
        float delta_y = rows - 1 - end_pts.y;
        end_pts.y = rows - 1;
        if (dir.y != 0)
            end_pts.x = end_pts.x + delta_y / dir.y * dir.x;
//        ROS_WARN("end_pts Y: (%f. %f)", end_pts.x, end_pts.y);
    }
}

void LineFeatureTracker::visualize_line(const Mat &imageMat1, const vector<Line> &lines, const string &name,
                                        const bool show_NMS_area) {
    //	Mat img_1;
    cv::Mat img1;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < lines.size(); ++k)
    {
        const Line& line1 = lines[k];
        const cv::Point2f& startPoint = line1.start_xy;
        const cv::Point2f& endPoint = line1.end_xy;

        if (show_NMS_area) {
            Point2f dir = endPoint - startPoint;
            float angle = atan2(dir.y, dir.x) / M_PI * 180.0;
            float len = norm((startPoint - endPoint));
//            ROS_DEBUG("image size: %u x %u", img1.cols, img1.rows);
//            ROS_DEBUG("start point: (%f, %f)", startPoint.x, startPoint.y);
//            ROS_DEBUG("end point: (%f, %f)", endPoint.x, endPoint.y);
            TicToc t_rect;
            DrawRotatedRectangle(img1, ((startPoint + endPoint) / 2.0), cv::Size(len + nms_extend, nms_extend), angle);
//            ROS_INFO("DrawRotatedRectangle costs: %fms", t_rect.toc());
        }
    }

    for (int k = 0; k < lines.size(); ++k) {

        const Line& line1 = lines[k];  // trainIdx
        const int& id = prev_lineID[k];

        unsigned int r = lowest + int(rand() % range);
        unsigned int g = lowest + int(rand() % range);
        unsigned int b = lowest + int(rand() % range);

        //line
        const cv::Point2f& startPoint = line1.start_xy;
        const cv::Point2f& endPoint = line1.end_xy;

//        if (line1.new_detect)
        if (id > last_feature_count) //new line in the forward frame
            cv::line(img1, startPoint, endPoint, cv::Scalar(0, 150, 255),2 ,8);
        else if (line1.updated_forwframe) //old lines updated in the forward frame (green)
            cv::line(img1, startPoint, endPoint, cv::Scalar(0, 255, 0),2 ,8);
        else if (line1.num_untracked >= LINE_MAX_UNTRACKED) // large num_untracked, to remove(red)
            cv::line(img1, startPoint, endPoint, cv::Scalar(0, 0, 255),2 ,8);
        else //old lines are not updated in the forward frame (blue)
            cv::line(img1, startPoint, endPoint, cv::Scalar(255, 0, 0),2 ,8);

        //start mid end point
        // b g r
        cv::circle(img1, startPoint, 1, cv::Scalar(255, 0, 0), 5);
//        cv::circle(img1, endPoint, 1, cv::Scalar(0, 0, 255), 5);
//        cv::circle(img1, 0.5 * startPoint + 0.5 * endPoint, 1, cv::Scalar(0, 255, 0), 5);
    }
    imshow(name, img1);
    waitKey(1);
}

void LineFeatureTracker::DrawRotatedRectangle(cv::Mat& image, const cv::Point2f& centerPoint, const cv::Size& rectangleSize,
                                              const float& rotationDegrees, const int& val)
{
    //    cv::Scalar color = cv::Scalar(255.0, 255.0, 255.0); // white

    // Create the rotated rectangle
    cv::RotatedRect rotatedRectangle(centerPoint, rectangleSize, rotationDegrees);

    // We take the edges that OpenCV calculated for us
    cv::Point2f vertices2f[4];
    rotatedRectangle.points(vertices2f);

//    vertices2f[0] = Point2f(0, 300);
//    vertices2f[1] = Point2f(300, 300);
//    vertices2f[2] = Point2f(300, 200);
//    vertices2f[3] = Point2f(0, 200);

//    cv::circle(image, vertices2f[0], 1, cv::Scalar(0, 0, 255), 5);
//    cv::circle(image, vertices2f[1], 1, cv::Scalar(0, 0, 255), 5);
//    cv::circle(image, vertices2f[2], 1, cv::Scalar(0, 0, 255), 5);
//    cv::circle(image, vertices2f[3], 1, cv::Scalar(0, 0, 255), 5);

//    {
//        // Convert them so we can use them in a fillConvexPoly
//        cv::Point vertices[4];
//        for (int i = 0; i < 4; ++i)
//            vertices[i] = vertices2f[i];
//
//        TicToc t_fill;
//        // Now we can fill the rotated rectangle with our specified color
//        cv::fillConvexPoly(image, vertices, 4, val);
//        ROS_DEBUG("cv::fillConvexPoly cost: %fms", t_fill.toc());
//    }

    {
        TicToc t_fill;
        uchar *buf = new uchar[4]{255, 255, 255, 0};
        fillRectangle(image, vertices2f, 4, buf);
//        ROS_DEBUG("fill Rectangle hand writing cost: %fms", t_fill.toc());
    }
}

void LineFeatureTracker::fillRectangle(Mat& img, const Point2f* pts, int npts, const void* color)
{
    assert(pts);
    assert(npts == 4); // only for rectangle

//    Size size = img.size();
    int pix_size = (int)img.elemSize();

    Point2f start_mid = 0.5 * (pts[0] + pts[3]);
    Point2f end_mid = 0.5 * (pts[1] + pts[2]);
    float threshold_p2l = norm(pts[0] - start_mid);

    //sort vertices by y value
    int y_min = pts[0].y, id_min = 0;
    for (int i = 0; i < npts; ++i)
        if (pts[i].y < y_min)
        {
            y_min = pts[i].y;
            id_min = i;
        }

//    Point pts_sorted[4];
//    int j = id_min;
//    for (int i = 0; i < 4; ++i)
//        pts_sorted[i] = pts[j % 4];

    //sort A B C D by y descending
    const Point2f& p_a = pts[id_min];
//    cv::circle(img, p_a, 1, cv::Scalar(255, 0, 0), 5);
    Point2f p_b = pts[(id_min + 1) % npts];
    Point2f p_c = pts[(id_min + npts - 1) % npts];
    if (p_b.y > p_c.y)
        swap(p_b, p_c);
//    cv::circle(img, p_b, 1, cv::Scalar(0, 255, 0), 5);
//    cv::circle(img, p_c, 1, cv::Scalar(0, 0, 255), 5);

    const Point2f& p_d = pts[(id_min + 2) % npts];
//    ROS_DEBUG("p_a: (%f, %f)", p_a.x, p_a.y);
//    ROS_DEBUG("p_b: (%f, %f)", p_b.x, p_b.y);
//    ROS_DEBUG("p_c: (%f, %f)", p_c.x, p_c.y);
//    ROS_DEBUG("p_d: (%f, %f)", p_d.x, p_d.y);
//    cv::circle(img, p_d, 1, cv::Scalar(255, 150, 0), 5);

    //angle = 0 or 90 degree
    if ((int)p_a.y == (int)p_b.y)
    {
        int row_start = (int)p_a.y;
        int row_target = (int)p_c.y;
        int col_left = (int)p_a.x;
        int col_right = (int)p_b.x;
        if (col_left > col_right)
            swap(col_left, col_right);

        //outside the image
        if (col_left >= img.cols || col_right < 0)
            return;
        if (col_left < 0)
            col_left = 0;
        if (col_right >= img.cols)
            col_right = img.cols - 1;
        for (int row_ptr = row_start; row_ptr <= row_target; ++row_ptr)
            for (int i = col_left; i <= col_right; ++i)
                img.at<cv::Vec3b>(row_ptr, i) = cv::Vec3b (255, 255, 255);
        return;
    }

    //set value between 2 lines
    uchar* image_ptr = img.ptr<uchar>();
    //delta x of line AB and AD, let AB be left, AC be right
    float delta_x_per_row_AB = (p_b.x - p_a.x) / (p_b.y - p_a.y);
//    ROS_DEBUG("delta_x_per_row_left: %f", delta_x_per_row_AB);
    float delta_x_per_row_AC = (p_c.x - p_a.x) / (p_c.y - p_a.y);
//    ROS_DEBUG("delta_x_per_row_right: %f", delta_x_per_row_AC);

    int row_start = (int)p_a.y;
    int row_target = (int)p_b.y;
    int row_ptr = row_start;
    float col_start_left = (int)p_a.x;
    float col_start_right = (int)p_a.x;
//    TicToc t_row;
    for (; row_ptr <= row_target; ++row_ptr)
    {
        col_start_left += delta_x_per_row_AB;
        if (row_ptr == row_target)
            col_start_left = p_b.x;
        col_start_right += delta_x_per_row_AC;

        //outside the image
        if (row_ptr < 0)//todo optimization
            continue;
        if (row_ptr >= img.rows)
            break;

        //set value by column
//        int delta_y = row_ptr - row_start;
        int col_left = col_start_left;
        int col_right = col_start_right;
        if (col_left > col_right)
            swap(col_left, col_right);
//        ROS_DEBUG("flii row: %d, %d --> %d", row_ptr, col_left, col_right);

        //outside the image
        if (col_left >= img.cols || col_right < 0)
            continue;
        if (col_left < 0)
            col_left = 0;
        if (col_right >= img.cols)
            col_right = img.cols - 1;
//        cv::circle(img, Point2f(col_left, row_ptr), 1, cv::Scalar(255, 150, 0), 5);
//        cv::circle(img, Point2f(col_right, row_ptr), 1, cv::Scalar(255, 150, 0), 5);

//        uchar* ptr = img.data;
//        ptr += img.step * row_ptr;
//        ICV_HLINE( ptr, col_left, col_right, color, pix_size );

        for (int i = col_left; i <= col_right; ++i)
        {
            img.at<cv::Vec3b>(row_ptr, i) = cv::Vec3b (255, 255, 255);
            //slower
//            img.at<cv::Vec3b>(row_ptr, i)[0] = 255;
//            img.at<cv::Vec3b>(row_ptr, i)[1] = 255;
//            img.at<cv::Vec3b>(row_ptr, i)[2] = 255;
        }
    }
//    ROS_DEBUG("set value row cost: %fms", t_row.toc());

    //between BD and AC
    row_start = row_ptr;
    row_target = (int)p_c.y;
    for (; row_ptr <= row_target; ++row_ptr)
    {
        col_start_left += delta_x_per_row_AC;
        col_start_right += delta_x_per_row_AC;
        if (row_ptr == row_target)
            col_start_right = p_c.x;

        //outside the image
        if (row_ptr < 0)//todo optimization
            continue;
        if (row_ptr >= img.rows)
            break;

        //set value by column
        int col_left = col_start_left;
        int col_right = col_start_right;
        if (col_left > col_right)
            swap(col_left, col_right);

        //outside the image
        if (col_left >= img.cols || col_right < 0)
            continue;
        if (col_left < 0)
            col_left = 0;
        if (col_right >= img.cols)
            col_right = img.cols - 1;

        for (int i = col_left; i <= col_right; ++i)
            img.at<cv::Vec3b>(row_ptr, i) = cv::Vec3b (255, 255, 255);
    }

    //between BD and CD
    row_start = row_ptr;
    row_target = (int)p_d.y;
    for (; row_ptr < row_target; ++row_ptr)
    {
        col_start_left += delta_x_per_row_AC;
        col_start_right += delta_x_per_row_AB;

        //outside the image
        if (row_ptr < 0)//todo optimization
            continue;
        if (row_ptr >= img.rows)
            break;

        //set value by column
        int col_left = col_start_left;
        int col_right = col_start_right;
        if (col_left > col_right)
            swap(col_left, col_right);

        //outside the image
        if (col_left >= img.cols || col_right < 0)
            continue;
        if (col_left < 0)
            col_left = 0;
        if (col_right >= img.cols)
            col_right = img.cols - 1;

        for (int i = col_left; i <= col_right; ++i)
            img.at<cv::Vec3b>(row_ptr, i) = cv::Vec3b (255, 255, 255);
    }
//    ROS_DEBUG("set value row cost: %fms", t_row.toc());
}
