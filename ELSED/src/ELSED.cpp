#include "ELSED.h"
#include "EdgeDrawer.h"

// Decides if we should take the image gradients as the interpolated version of the pixels right in the segment
// or if they are ready directly from the image
#define UPM_SD_USE_REPROJECTION

namespace upm {

ELSED::ELSED(const ELSEDParams &params) : params(params) {
}

Segments ELSED::detect(const cv::Mat &image) {
  processImage(image);
  // std::cout << "ELSED detected: " << segments.size() << " segments" << std::endl;
  return segments;
}

SalientSegments ELSED::detectSalient(const cv::Mat &image) {
  processImage(image);
  // std::cout << "ELSED detected: " << salientSegments.size() << " salient segments" << std::endl;
  return salientSegments;
}

ImageEdges ELSED::detectEdges(const cv::Mat &image) {
  processImage(image);
  return getSegmentEdges();
}

const LineDetectionExtraInfo &ELSED::getImgInfo() const {
  return *imgInfo;
}

void ELSED::processImage(const cv::Mat &_image) {
//    std::cout << "process image \n";
    // Check that the image is a grayscale image
  cv::Mat image;
  switch (_image.channels()) {
    case 3:
      cv::cvtColor(_image, image, cv::COLOR_BGR2GRAY);
      break;
    case 4:
      cv::cvtColor(_image, image, cv::COLOR_BGRA2GRAY);
      break;
    default:
      image = _image;
      break;
  }
  assert(image.channels() == 1);
  // Clear previous state
  this->clear();

  if (image.empty()) {
    return;
  }

  // Set the global image
  // Filter the image
  if (params.ksize > 2) {
    cv::GaussianBlur(image, blurredImg, cv::Size(params.ksize, params.ksize), params.sigma);
  } else {
    blurredImg = image;
  }

  // Compute the input image derivatives
  imgInfo = computeGradients(blurredImg, params.gradientThreshold);

  bool anchoThIsZero;
  uint8_t anchorTh = params.anchorThreshold;
  do {
    anchoThIsZero = anchorTh == 0;
    // Detect edges and segment in the input image
    computeAnchorPoints(imgInfo->dirImg,
                        imgInfo->gImgWO,
                        imgInfo->gImg,
                        nmsImg,
                        params.scanIntervals,//todo
                        anchorTh,
                        anchors);

    // If we couldn't find any anchor, decrease the anchor threshold
    if (anchors.empty()) {
      // std::cout << "Cannot find any anchor with AnchorTh = " << int(anchorTh)
      //      << ", repeating process with AnchorTh = " << (anchorTh / 2) << std::endl;
      anchorTh /= 2;
    }

  } while (anchors.empty() && !anchoThIsZero);
  // LOGD << "Detected " << anchors.size() << " anchor points ";
  edgeImg = cv::Mat::zeros(imgInfo->imageHeight, imgInfo->imageWidth, CV_8UC1);
  nmsImg = cv::Mat::zeros(imgInfo->imageHeight, imgInfo->imageWidth, CV_8UC1);

    drawer = std::make_shared<EdgeDrawer>(imgInfo,
                                          edgeImg,
                                          nmsImg,
                                          params.lineFitErrThreshold,
                                          params.pxToSegmentDistTh,
                                          params.minLineLen,
                                          params.treatJunctions,
                                          params.listJunctionSizes,
                                          params.junctionEigenvalsTh,
                                          params.junctionAngleTh);

    if (linesExist.empty()) {
        drawAnchorPoints(imgInfo->dirImg.ptr(), nmsImg.ptr(), anchors, edgeImg.ptr());
        outside_nms_lines = segments;
        return;
    }

    //set NMS by all exist lines
//    setLinesNMSArea();
    //划分anchors到的NMS区域
    std::vector <Pixel> outside_nms_anchors;
    cv::Mat nms_all;
    nms_all = cv::Mat::zeros(imgInfo->imageHeight, imgInfo->imageWidth, CV_8UC1);
    setLinesNMSArea(nms_all);
    filterAnchorsInsideNMSArea(nms_all, anchors, inside_nms_anchors, outside_nms_anchors);
    std::cout << "anchors: " << anchors.size() << ", inside NMS area: " << inside_nms_anchors.size()
              << ", outside: " << outside_nms_anchors.size() << std::endl;

    extractLinesinsideNMS();
    segments.clear();
    nmsImg = nms_all;
    //仅提取NMS外侧点
    drawer->setOnlyInsideNMSRegion(false);
    one_line_inside_NMS_area = true;
//    nmsImg = cv::Mat::zeros(imgInfo->imageHeight, imgInfo->imageWidth, CV_8UC1);
//    edgeImg = cv::Mat::zeros(imgInfo->imageHeight, imgInfo->imageWidth, CV_8UC1);
    drawAnchorPoints(imgInfo->dirImg.ptr(), nmsImg.ptr(), outside_nms_anchors, edgeImg.ptr());
    outside_nms_lines = segments;
//    std::string file_name;
//    file_name = "/tmp/nmsImg_final.jpg";
//    cv::imwrite(file_name, nmsImg);
//    file_name = "/tmp/edgeImg_final.jpg";
//    cv::imwrite(file_name, edgeImg);

}

LineDetectionExtraInfoPtr ELSED::computeGradients(const cv::Mat &srcImg, short gradientTh) {
  LineDetectionExtraInfoPtr dstInfo = std::make_shared<LineDetectionExtraInfo>();
  cv::Sobel(srcImg, dstInfo->dxImg, CV_16SC1, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
  cv::Sobel(srcImg, dstInfo->dyImg, CV_16SC1, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);

  int nRows = srcImg.rows;
  int nCols = srcImg.cols;
  int i;

  dstInfo->imageWidth = srcImg.cols;
  dstInfo->imageHeight = srcImg.rows;
  dstInfo->gImgWO = cv::Mat(srcImg.size(), dstInfo->dxImg.type());
  dstInfo->gImg = cv::Mat(srcImg.size(), dstInfo->dxImg.type());
  dstInfo->dirImg = cv::Mat(srcImg.size(), CV_8UC1);

  const int16_t *pDX = dstInfo->dxImg.ptr<int16_t>();
  const int16_t *pDY = dstInfo->dyImg.ptr<int16_t>();
  auto *pGr = dstInfo->gImg.ptr<int16_t>();
  auto *pGrWO = dstInfo->gImgWO.ptr<int16_t>();
  auto *pDir = dstInfo->dirImg.ptr<uchar>();
  int16_t abs_dx, abs_dy, sum;
  const int totSize = nRows * nCols;
  for (i = 0; i < totSize; ++i) {
    // Absolute value
    abs_dx = UPM_ABS(pDX[i]);
    // Absolute value
    abs_dy = UPM_ABS(pDY[i]);
    sum = abs_dx + abs_dy;
    // Divide by 2 the gradient
    pGrWO[i] = sum;
    pGr[i] = sum < gradientTh ? 0 : sum;
    // Select between vertical or horizontal gradient
    pDir[i] = abs_dx >= abs_dy ? UPM_EDGE_VERTICAL : UPM_EDGE_HORIZONTAL;
  }

  return dstInfo;
}

inline void ELSED::computeAnchorPoints(const cv::Mat &dirImage,
                                       const cv::Mat &gradImageWO,
                                       const cv::Mat &gradImage,
                                       const cv::Mat &nmsImage,
                                       int scanInterval,
                                       int anchorThresh,
                                       std::vector<Pixel> &anchorPoints) {  // NOLINT

  int imageWidth = gradImage.cols;
  int imageHeight = gradImage.rows;

  // Get pointers to the thresholded gradient image and to the direction image
  const auto *gradImg = gradImage.ptr<int16_t>();
  const auto *dirImg = dirImage.ptr<uint8_t>();
//    const auto *nmsImg = nmsImage.ptr<uint8_t>();

    // Extract the anchors in the gradient image, store into a vector
  unsigned int pixelNum = imageWidth * imageHeight;
  unsigned int edgePixelArraySize = pixelNum / (2.5 * scanInterval);
  anchorPoints.resize(edgePixelArraySize);

  int nAnchors = 0;
  int indexInArray;
  unsigned int w, h;
  for (w = 1; w < imageWidth - 1; w += scanInterval) {
    for (h = 1; h < imageHeight - 1; h += scanInterval) {
      indexInArray = h * imageWidth + w;

      // If there is no gradient in the pixel avoid the anchor generation
      if (gradImg[indexInArray] == 0) continue;
//      if (nmsImg[indexInArray] == UPM_ED_NMS_PX) continue;

      // To be an Anchor the pixel must have a gradient magnitude
      // anchorThreshold_ units higher than that of its neighbours
      if (dirImg[indexInArray] == UPM_EDGE_HORIZONTAL) {
        // Check if (w, h) is accepted as an anchor using the Anchor Threshold.
        // We compare with the top and bottom pixel gradients
        if (gradImg[indexInArray] >= gradImg[indexInArray - imageWidth] + anchorThresh &&
            gradImg[indexInArray] >= gradImg[indexInArray + imageWidth] + anchorThresh) {
          anchorPoints[nAnchors].x = w;
          anchorPoints[nAnchors].y = h;
          nAnchors++;
        }
      } else {
        // Check if (w, h) is accepted as an anchor using the Anchor Threshold.
        // We compare with the left and right pixel gradients
        if (gradImg[indexInArray] >= gradImg[indexInArray - 1] + anchorThresh &&
            gradImg[indexInArray] >= gradImg[indexInArray + 1] + anchorThresh) {
          anchorPoints[nAnchors].x = w;
          anchorPoints[nAnchors].y = h;
          nAnchors++;
        }
      }
    }
  }
  anchorPoints.resize(nAnchors);
}

    void ELSED::addAnchors(const cv::Mat &dirImage,
                    const cv::Mat &gradImageWO,
                    const cv::Mat &gradImage,
                    const cv::Mat &nmsImage,
                    int scanInterval,
                    int anchorThresh,
                           std::vector<cv::Point2f> candidates,
                           std::vector<Pixel> &anchorPoints)
    {
        int imageWidth = gradImage.cols;
        int imageHeight = gradImage.rows;

        // Get pointers to the thresholded gradient image and to the direction image
        const auto *gradImg = gradImage.ptr<int16_t>();
        const auto *dirImg = dirImage.ptr<uint8_t>();
//    const auto *nmsImg = nmsImage.ptr<uint8_t>();

        // Extract the anchors in the gradient image, store into a vector
        unsigned int pixelNum = imageWidth * imageHeight;
//        unsigned int edgePixelArraySize = pixelNum / (2.5 * scanInterval);
//        anchorPoints.resize(edgePixelArraySize);

        anchorPoints.clear();
//        int nAnchors = 0;
        int indexInArray;
        unsigned int w, h;
        for (int i = 0; i < candidates.size(); ++i) {
            h = (unsigned int)candidates[i].y;
            w = (unsigned int)candidates[i].x;
            indexInArray = h * imageWidth + w;

            // If there is no gradient in the pixel avoid the anchor generation
//            std::cout << "gradImg = " << gradImg[indexInArray] <<std::endl;
            if (gradImg[indexInArray] == 0) continue;
//      if (nmsImg[indexInArray] == UPM_ED_NMS_PX) continue;

            // To be an Anchor the pixel must have a gradient magnitude
            // anchorThreshold_ units higher than that of its neighbours
            if (dirImg[indexInArray] == UPM_EDGE_HORIZONTAL) {
                // Check if (w, h) is accepted as an anchor using the Anchor Threshold.
                // We compare with the top and bottom pixel gradients
                if (gradImg[indexInArray] >= gradImg[indexInArray - imageWidth] + anchorThresh &&
                    gradImg[indexInArray] >= gradImg[indexInArray + imageWidth] + anchorThresh) {
                    anchorPoints.emplace_back(Pixel(w, h));
//                    anchorPoints[nAnchors].x = w;
//                    anchorPoints[nAnchors].y = h;
//                    nAnchors++;
//                    std::cout << "add as UPM_EDGE_HORIZONTAL\n";
                }
            } else {
                // Check if (w, h) is accepted as an anchor using the Anchor Threshold.
                // We compare with the left and right pixel gradients
                if (gradImg[indexInArray] >= gradImg[indexInArray - 1] + anchorThresh &&
                    gradImg[indexInArray] >= gradImg[indexInArray + 1] + anchorThresh) {
                    anchorPoints.emplace_back(Pixel(w, h));
//                    anchorPoints[nAnchors].x = w;
//                    anchorPoints[nAnchors].y = h;
//                    nAnchors++;
//                    std::cout << "add as UPM_EDGE_VERTICAL\n";
                }
            }
        }
//    std::cout << "sample points: "<< candidates.size() <<"  anchors_new: "<< anchorPoints.size()<<std::endl;
    }

void ELSED::clear() {
  imgInfo = nullptr;
  edges.clear();
  segments.clear();
  salientSegments.clear();
  anchors.clear();
  drawer = nullptr;
  blurredImg = cv::Mat();
  edgeImg = cv::Mat();
  nmsImg = cv::Mat();
}

inline int calculateNumPtsToTrim(int nPoints) {
  return std::min(5.0, nPoints * 0.1);
}

// Linear interpolation. s is the starting value, e the ending value
// and t the point offset between e and s in range [0, 1]
inline float lerp(float s, float e, float t) { return s + (e - s) * t; }

// Bi-linear interpolation of point (tx, ty) in the cell with corner values [[c00, c01], [c10, c11]]
inline float blerp(float c00, float c10, float c01, float c11, float tx, float ty) {
  return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

void ELSED::drawAnchorPoints(const uint8_t *dirImg,
                             const uint8_t *pNMSImg,
                             const std::vector<Pixel> &anchorPoints,
                             uint8_t *pEdgeImg) {
  assert(imgInfo && imgInfo->imageWidth > 0 && imgInfo->imageHeight > 0);
  assert(!imgInfo->gImg.empty() && !imgInfo->dirImg.empty() && pEdgeImg);
  assert(drawer);
  assert(!edgeImg.empty());

  int imageWidth = imgInfo->imageWidth;
  int imageHeight = imgInfo->imageHeight;
  bool expandHorizontally;
  int indexInArray;
  unsigned char lastDirection;  // up = 1, right = 2, down = 3, left = 4;

  if (anchorPoints.empty()) {
    // No anchor points detected in the image
    return;
  }

  const double validationTh = params.validationTh;

  //遍历所有anchor point
  for (const auto &anchorPoint: anchorPoints) {
    // LOGD << "Managing new Anchor point: " << anchorPoint;
    indexInArray = anchorPoint.y * imageWidth + anchorPoint.x;

    if (pEdgeImg[indexInArray]) {
      // If anchor i is already been an edge pixel
      continue;
    }

//      if (pNMSImg[indexInArray] == UPM_ED_NMS_PX) {
//          // If anchor i is already been an edge pixel
//          continue;
//      }

    // If the direction of this pixel is horizontal, then go left and right.
    expandHorizontally = dirImg[indexInArray] == UPM_EDGE_HORIZONTAL;

    /****************** First side Expanding (Right or Down) ***************/
    // Select the first side towards we want to move. If the gradient points
    // horizontally select the right direction and otherwise the down direction.
    lastDirection = expandHorizontally ? UPM_RIGHT : UPM_DOWN;

    drawer->drawEdgeInBothDirections(lastDirection, anchorPoint);
  }

  ////检测segments有效性
  double theta, angle;
  float saliency;
  bool valid; //true: inliers > outliers
  int endpointDist, nOriInliers, nOriOutliers;
  int nInside_NMS, nInside_pxs;
#ifdef UPM_SD_USE_REPROJECTION
  cv::Point2f p;
  float lerp_dx, lerp_dy;
  int x0, y0, x1, y1;
#endif
  int16_t *pDx = imgInfo->dxImg.ptr<int16_t>();
  int16_t *pDy = imgInfo->dyImg.ptr<int16_t>();
  segments.reserve(drawer->getDetectedFullSegments().size());
  salientSegments.reserve(drawer->getDetectedFullSegments().size());
  full_segments_id.reserve(drawer->getDetectedFullSegments().size()); //valid segments in drawer->getDetectedFullSegments()

    cv::Mat nms_region;
    if (one_line_inside_NMS_area)
    {
        nms_region = cv::Mat::zeros(imgInfo->imageHeight, imgInfo->imageWidth, CV_8UC1); // new line NMS region
        uint8_t *pNMS_region = nms_region.ptr();
    }

    //compute segments length and sort segments by length
    for (FullSegmentInfo &detectedSeg: drawer->getDetectedFullSegments())
        detectedSeg.computeLineInfo();
    std::sort(drawer->getDetectedFullSegments().begin(), drawer->getDetectedFullSegments().end(),
              [&](const FullSegmentInfo &s1, const FullSegmentInfo &s2) { return s1.length > s2.length; });
//    for (FullSegmentInfo &detectedSeg: drawer->getDetectedFullSegments())
//        std::cout<<"length : "<< detectedSeg.length<<std::endl;

//    std::cout<<"drawer->getDetectedFullSegments().size() : "<< drawer->getDetectedFullSegments().size()<<std::endl;

    //  for (const FullSegmentInfo &detectedSeg: drawer->getDetectedFullSegments())
    for (int i = 0; i < drawer->getDetectedFullSegments().size(); ++i) {
        const FullSegmentInfo &detectedSeg = drawer->getDetectedFullSegments()[i];

    valid = true;
    if (params.validate) {
      if (detectedSeg.getNumOfPixels() < 2) {
        valid = false;
      } else {
        // Get the segment angle
        Segment s = detectedSeg.getEndpoints();
        theta = segAngle(s) + M_PI_2; // line angle
        // Force theta to be in range [0, M_PI)
        while (theta < 0) theta += M_PI;
        while (theta >= M_PI) theta -= M_PI;

        // Calculate the line equation as the cross product os the endpoints
        cv::Vec3f l = cv::Vec3f(s[0], s[1], 1).cross(cv::Vec3f(s[2], s[3], 1));
        // Normalize the line direction
        l /= std::sqrt(l[0] * l[0] + l[1] * l[1]);
        cv::Point2f perpDir(l[0], l[1]);

        // For each pixel in the segment compute its angle
        int nPixelsToTrim = calculateNumPtsToTrim(detectedSeg.getNumOfPixels());

        Pixel firstPx = detectedSeg.getFirstPixel();
        Pixel lastPx = detectedSeg.getLastPixel();

        nOriInliers = 0;
        nOriOutliers = 0;
          nInside_NMS = 0;
          nInside_pxs = 0;
          std::unordered_map<int, int> NMS_map; //(nms_region_id, num_inside)
        for (auto px: detectedSeg) {

          // If the point is not an inlier avoid it
          if (edgeImg.at<uint8_t>(px.y, px.x) != UPM_ED_SEGMENT_INLIER_PX) {
            continue;
          }

            // If the point is not a free pixel, avoid it
            if (one_line_inside_NMS_area)
            {
                const uint8_t &region_id = nms_region.at<uint8_t>(px.y, px.x); // detectedSeg id + 1
                if (region_id != 0) {
                    ++nInside_NMS;
                    ++NMS_map[region_id - 1];
                }
            }

          endpointDist = detectedSeg.horizontal() ?
                         std::min(std::abs(px.x - lastPx.x), std::abs(px.x - firstPx.x)) :
                         std::min(std::abs(px.y - lastPx.y), std::abs(px.y - firstPx.y));

          if (endpointDist < nPixelsToTrim) {
            continue;
          }

#ifdef false
          // Re-project the point into the segment. To do this, we should move pixel.dot(l)
          // units (the distance between the pixel and the segment) in the direction
          // perpendicular to the segment (perpDir).
          p = cv::Point2f(px.x, px.y) - perpDir * cv::Vec3f(px.x, px.y, 1).dot(l);
          // Get the values around the point p to do the bi-linear interpolation
          x0 = p.x < 0 ? 0 : p.x;
          if (x0 >= imageWidth) x0 = imageWidth - 1;
          y0 = p.y < 0 ? 0 : p.y;
          if (y0 >= imageHeight) y0 = imageHeight - 1;
          x1 = p.x + 1;
          if (x1 >= imageWidth) x1 = imageWidth - 1;
          y1 = p.y + 1;
          if (y1 >= imageHeight) y1 = imageHeight - 1;
          //Bi-linear interpolation of Dx and Dy
          lerp_dx = blerp(pDx[y0 * imageWidth + x0], pDx[y0 * imageWidth + x1],
                          pDx[y1 * imageWidth + x0], pDx[y1 * imageWidth + x1],
                          p.x - int(p.x), p.y - int(p.y));
          lerp_dy = blerp(pDy[y0 * imageWidth + x0], pDy[y0 * imageWidth + x1],
                          pDy[y1 * imageWidth + x0], pDy[y1 * imageWidth + x1],
                          p.x - int(p.x), p.y - int(p.y));
          // Get the gradient angle
          angle = std::atan2(lerp_dy, lerp_dx);
#else
          indexInArray = px.y * imageWidth + px.x;
          angle = std::atan2(pDy[indexInArray], pDx[indexInArray]); //pixel gradient angel
#endif
          // Force theta to be in range [0, M_PI)
          if (angle < 0) angle += M_PI;
          if (angle >= M_PI) angle -= M_PI;
          circularDist(theta, angle, M_PI) > validationTh ? nOriOutliers++ : nOriInliers++;

          ++nInside_pxs;
        }

        valid = nOriInliers > nOriOutliers;
        saliency = nOriInliers;

        if (one_line_inside_NMS_area)
          {
//              std::cout << "nInside_NMS : " << nInside_NMS << "   nInside_pxs:" << nInside_pxs << std::endl;
              if (valid && nInside_NMS > 0.3 * nInside_pxs) {
                  valid = false;
              }
          }
      }
    } else {
      saliency = segLength(detectedSeg.getEndpoints());
    }

    //valid == true: nOriInliers > nOriOutliers
    if (valid) {
      const Segment &endpoints = detectedSeg.getEndpoints();
      segments.push_back(endpoints);
      salientSegments.emplace_back(endpoints, saliency);

      full_segments_id.emplace_back(i);
      //add new NMS area
        if (one_line_inside_NMS_area)
            DrawRotatedRectangle(nms_region, cv::Point2f(detectedSeg.center_x, detectedSeg.center_y), cv::Size(detectedSeg.length, params.nms_height_extend),
                           detectedSeg.line_angle / M_PI * 180.0, i + 1);
    }
  }

////  //todo
////  // 通过NMS区域筛选线特征
////  // 按照长度排序后，逐一添加NMS
//    bool sortByLength = true;
//    if (sortByLength)
//        std::sort(segments.begin(), segments.end(), compareLength);
//
//    std::vector<length_id> length_segments;
//    length_segments.reserve(segments.size());
//    for (int i = 0; i < segments.size(); ++i) {
//        length_segments[i].segments_id = i;
//        length_segments[i].global_segments_id = full_segments_id[i];
//        Segment& s1 = segments[i];
//        length_segments[i].length = (s1(0) - s1(2)) * (s1(0) - s1(2)) + (s1(1) - s1(3)) * (s1(1) - s1(3));
//    }
//    std::sort(length_segments.begin(), length_segments.end(),
//              [&](const length_id& l1, const length_id& l2)->bool {return l1.length > l2.length;});
//
//
//    bool check_NMS = true; // 新线特征检查NMS重叠清空，防止线特征太密集
//    if (check_NMS)
//    {
//        cv::Mat nms_region = cv::Mat::zeros(imgInfo->imageHeight, imgInfo->imageWidth, CV_8UC1); // new line NMS region
//        uint8_t *pNMS_region = nms_region.ptr();
//
//        for (int i = 0; i < length_segments.size(); ++i) {
//            const int& global_id = length_segments[i].global_segments_id;
//            const FullSegmentInfo &detectedSeg = drawer->getDetectedFullSegments()[global_id];
//
//            int num_NMS_px = 0;
//            for (auto px: detectedSeg) {
//
//                int indexInArray = px.y * imageWidth + px.x;
//                // If the point is not an inlier avoid it
//                if (pNMS_region[indexInArray] == UPM_ED_NMS_PX) {
//                    ++num_NMS_px;
//                }
//
//                endpointDist = detectedSeg.horizontal() ?
//                               std::min(std::abs(px.x - lastPx.x), std::abs(px.x - firstPx.x)) :
//                               std::min(std::abs(px.y - lastPx.y), std::abs(px.y - firstPx.y));
//
//                if (endpointDist < nPixelsToTrim) {
//                    continue;
//                }
//
//                indexInArray = px.y * imageWidth + px.x;
//                angle = std::atan2(pDy[indexInArray], pDx[indexInArray]); //pixel gradient angel
//
//                // Force theta to be in range [0, M_PI)
//                if (angle < 0) angle += M_PI;
//                if (angle >= M_PI) angle -= M_PI;
//                circularDist(theta, angle, M_PI) > validationTh ? nOriOutliers++ : nOriInliers++;
//            }
//
//        }
//    }


//    cv::imwrite("/home/hk/ws_pl-lvi-sam/setLinesExist_NMS_5.jpg", edgeImg);
//
//    setLinesNMSArea();
//  int j = 0;
//  std::cout<<"*********segments.size(): "<< segments.size()<<std::endl;
//    int startIndexInArray, endIndexInArray;
//    for (int i = 0; i < segments.size(); ++i) {
//        const cv::Vec4f& endpoints = segments[i];
//        startIndexInArray = endpoints(1) * imageWidth + endpoints(0);//start point
//        endIndexInArray = endpoints(3) * imageWidth + endpoints(2);//start point
//
//        if (pNMSImg[startIndexInArray] == UPM_ED_NMS_PX || pNMSImg[endIndexInArray] == UPM_ED_NMS_PX) {
//            // If anchor i is already been an edge pixel
//            continue;
//        }
//        segments[j++] = segments[i];
//    }
//    segments.resize(j);
//    std::cout<<"*********segments.size(): "<< segments.size()<<std::endl;
////    cv::imwrite("/home/hk/ws_pl-lvi-sam/setLinesExist_NMS_6.jpg", edgeImg);


}

ImageEdges ELSED::getAllEdges() const {
  assert(drawer);
  ImageEdges result;
  for (const FullSegmentInfo &s: drawer->getDetectedFullSegments())
    result.push_back(s.getPixels());
  return result;
}

ImageEdges ELSED::getSegmentEdges() const {
  assert(drawer);
  ImageEdges result;
  for (const FullSegmentInfo &s: drawer->getDetectedFullSegments())
    result.push_back(s.getPixels());
  return result;
}

const LineDetectionExtraInfoPtr &ELSED::getImgInfoPtr() const {
  return imgInfo;
}

void ELSED::DrawRotatedRectangle(cv::Mat& image, const cv::Point2f& centerPoint, const cv::Size& rectangleSize,
                                 const float& rotationDegrees, const int& val)
{
//    cv::Scalar color = cv::Scalar(255.0, 255.0, 255.0); // white

    // Create the rotated rectangle
    cv::RotatedRect rotatedRectangle(centerPoint, rectangleSize, rotationDegrees);

    // We take the edges that OpenCV calculated for us
    cv::Point2f vertices2f[4];
    rotatedRectangle.points(vertices2f);

    // Convert them so we can use them in a fillConvexPoly
    cv::Point vertices[4];
    for(int i = 0; i < 4; ++i){
        vertices[i] = vertices2f[i];
    }

    // Now we can fill the rotated rectangle with our specified color
    cv::fillConvexPoly(image,
                       vertices,
                       4,
                       val);
}

void ELSED::setLinesExist(const Segments& linesExist) {
//    auto *pEdge = edgeImg.ptr<uint8_t>();
//    const int totSize = edgeImg.rows * edgeImg.cols / 2;
//    for (int i = 0; i < totSize; ++i) {
//        pEdge[i] = UPM_ED_EDGE_PIXEL;
//    }
}

void ELSED::setLinesNMSArea() {
    const int line_size = linesExist.size();
    for (int i = 0; i < line_size; ++i) {
        lineInfo &line = linesExist[i];
        DrawRotatedRectangle(nmsImg, line.centerPoint, cv::Size(line.length + params.nms_width_extend, params.nms_height_extend),
                             line.angle);
    }
}

void ELSED::setLinesNMSArea(cv::Mat& image) {
    const int line_size = linesExist.size();
    for (int i = 0; i < line_size; ++i) {
        lineInfo &line = linesExist[i];
        DrawRotatedRectangle(image, line.centerPoint, cv::Size(line.length + params.nms_width_extend, params.nms_height_extend),
                             line.angle);
    }
}

void ELSED::setAnchorsExist()
{
    const int line_size = linesExist.size();
    anchorsExist.resize(2 * line_size);
    for (int i = 0; i < line_size; ++i) {
        lineInfo &line = linesExist[i];

        anchorsExist[2 * i].x = line.startPoint.x;
        anchorsExist[2 * i].y = line.startPoint.y;
        anchorsExist[2 * i + 1].x = line.endPoint.x;
        anchorsExist[2 * i + 1].y = line.endPoint.y;
    }
}

void ELSED::filterAnchorsInsideNMSRegion(std::vector<Pixel> &anchorPoints)
{
    int imageWidth = imgInfo->imageWidth;
    int imageHeight = imgInfo->imageHeight;
    int indexInArray;
    uint8_t *pNMSImg = nmsImg.ptr();

    //filter anchor points
    anchorPoints.resize(anchors.size());
    int j = 0;
    for (int i = 0; i < anchors.size(); ++i) {
        const Pixel& anchorPoint = anchors[i];
        indexInArray = anchorPoint.y * imageWidth + anchorPoint.x;
        if (pNMSImg[indexInArray])
            anchors[j++] = anchors[i];
    }
    anchors.resize(j);
}

void ELSED::filterAnchorsInsideNMSArea(const std::vector<Pixel> &anchors_input, std::vector<std::vector<Pixel>> &inside_nms_anchors,
                                       std::vector<Pixel> &outside_nms_anchors)
{
    int imageWidth = imgInfo->imageWidth;
    int imageHeight = imgInfo->imageHeight;
    int indexInArray;

    uint8_t *pNMSImg = nmsImg.ptr();

    //filter anchor points
    int NMS_region_size = linesExist.size();
    outside_nms_anchors.resize(anchors.size());
    inside_nms_anchors.resize(NMS_region_size);
    int j = 0;
    for (int i = 0; i < anchors.size(); ++i) {
        const Pixel& anchorPoint = anchors[i];
        indexInArray = anchorPoint.y * imageWidth + anchorPoint.x;
//        std::cout<<"indexInArray:"<< indexInArray <<std::endl;
        int nms_id = pNMSImg[indexInArray];
//        std::cout<<"nms_id:"<< nms_id <<std::endl;
        if (nms_id)
        {
            inside_nms_anchors[nms_id - 1].emplace_back(anchors[i]);
//            std::cout<<"nms_id:"<< nms_id <<std::endl;
        }
        else
            outside_nms_anchors[j++] = anchorPoint;
    }
    outside_nms_anchors.resize(j);
}

//todo 整行赋值内存加速
//     验证采样点是否为anchor
//     grad阈值调整
void ELSED::fillRectangle(cv::Mat& img, const cv::Point2f* pts, const int& npts, const uint8_t& color)
{
    assert(pts);
    assert(npts == 4); // only for rectangle

//    cv::Size size = img.size();
    int pix_size = (int)img.elemSize();

    cv::Point2f start_mid = 0.5 * (pts[0] + pts[3]);
    cv::Point2f end_mid = 0.5 * (pts[1] + pts[2]);
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
    const cv::Point2f& p_a = pts[id_min];
//    cv::circle(img, p_a, 1, cv::Scalar(255, 0, 0), 5);
    cv::Point2f p_b = pts[(id_min + 1) % npts];
    cv::Point2f p_c = pts[(id_min + npts - 1) % npts];
    if (p_b.y > p_c.y)
        swap(p_b, p_c);
//    cv::circle(img, p_b, 1, cv::Scalar(0, 255, 0), 5);
//    cv::circle(img, p_c, 1, cv::Scalar(0, 0, 255), 5);

    const cv::Point2f& p_d = pts[(id_min + 2) % npts];
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
            std::swap(col_left, col_right);

        //outside the image
        if (col_left >= img.cols || col_right < 0)
            return;
        if (col_left < 0)
            col_left = 0;
        if (col_right >= img.cols)
            col_right = img.cols - 1;
        for (int row_ptr = row_start; row_ptr <= row_target; ++row_ptr)
            for (int i = col_left; i <= col_right; ++i)
                img.at<uint8_t>(row_ptr, i) = color;
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
            std::swap(col_left, col_right);
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
            img.at<uint8_t>(row_ptr, i) = color;
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
            std::swap(col_left, col_right);

        //outside the image
        if (col_left >= img.cols || col_right < 0)
            continue;
        if (col_left < 0)
            col_left = 0;
        if (col_right >= img.cols)
            col_right = img.cols - 1;

        for (int i = col_left; i <= col_right; ++i)
            img.at<uint8_t>(row_ptr, i) = color;
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
            std::swap(col_left, col_right);

        //outside the image
        if (col_left >= img.cols || col_right < 0)
            continue;
        if (col_left < 0)
            col_left = 0;
        if (col_right >= img.cols)
            col_right = img.cols - 1;

        for (int i = col_left; i <= col_right; ++i)
            img.at<uint8_t>(row_ptr, i) = color;
    }
}

    void ELSED::extractLinesinsideNMS()
    {
        const bool cout_debug = false;
        if (cout_debug)
            std::cout << "extractLinesinsideNMS\n";

        if (linesExist.empty())
            return;

        int imageWidth = imgInfo->imageWidth;
        int imageHeight = imgInfo->imageHeight;
        int indexInArray;

        uint8_t *pNMSImg = nmsImg.ptr();
        auto *pGr = imgInfo->gImg.ptr<int16_t>();

        const int line_size = linesExist.size();
        nms_lines.resize(line_size);

        drawer->setOnlyInsideNMSRegion(true);
        one_line_inside_NMS_area = false;

        //backtrack
        for (int i = 0; i < line_size; ++i)
        {
            if (cout_debug)
                std::cout << "i = "<< i <<"\n";

            lineInfo &line = linesExist[i];
            if (!line.need_detect)
                continue;

            //1. set NMS area for a line
            // Create the rotated rectangle
            cv::RotatedRect rotatedRectangle(line.centerPoint, cv::Size(line.length + params.nms_width_extend, params.nms_height_extend), line.angle);
            // We take the edges that OpenCV calculated for us
            cv::Point2f vertices2f[4];
            rotatedRectangle.points(vertices2f);
            // Convert them so we can use them in a fillConvexPoly
            cv::Point vertices[4];
            for(int j = 0; j < 4; ++j)
                vertices[j] = vertices2f[j];
            // Now we can fill the rotated rectangle with our specified color
            cv::fillConvexPoly(nmsImg, vertices, 4, UPM_ED_NMS_PX);
//            fillRectangle(nmsImg, vertices2f, 4, UPM_ED_NMS_PX);

            //2. filter anchors inside the NMS area by point to line distance
            std::vector<Pixel> anchors_filtered;
//            cv::Mat tmp;
//            nmsImg.copyTo(tmp);
            for (int j = 0; j < inside_nms_anchors.size(); ++j)
            {
                const Pixel& anchorPoint = inside_nms_anchors[j];
                indexInArray = anchorPoint.y * imageWidth + anchorPoint.x;
//                int nms_id = pNMSImg[indexInArray];
//                if (pNMSImg[indexInArray] == UPM_ED_NMS_PX) {
                if (pNMSImg[indexInArray]) {
                    anchors_filtered.emplace_back(inside_nms_anchors[j]);
//                    cv::Point2f anchor(anchorPoint.x, anchorPoint.y);
//                    cv::circle(tmp, anchor,3,255);
//                    std::cout<< "anchor : (" << anchor.x <<", "<< anchor.y << ")\n";
                }
            }

            std::vector<Pixel> anchors_new;
            addAnchors(imgInfo->dirImg,
                       imgInfo->gImgWO,
                       imgInfo->gImg,
                       nmsImg,
                       params.scanIntervals,//todo
                       params.anchorThreshold,
                       line.sample_points,
                       anchors_new);
//            for (int j = 0; j < anchors_new.size(); ++j) {
//                cv::Point2f anchor(anchors_new[j].x, anchors_new[j].y);
//                cv::circle(tmp, anchor,3,255);
//            }
            if (!anchors_new.empty())
                anchors_filtered.insert(anchors_filtered.end(), anchors_new.begin(), anchors_new.end());

//            std::string file_name;
//            file_name = "/tmp/nms_img_" + std::to_string(i) + ".jpg";
//            cv::imwrite(file_name, tmp);

            if (cout_debug)
                std::cout<< i << " anchors_filtered : " << anchors_filtered.size()<< std::endl;

            //todo save image for testing
//            std::cout << "save edge images\n";
//            std::string file_name;
//            file_name = "/tmp/before_drawAnchorPoints_" + std::to_string(i) + ".jpg";
//            cv::imwrite(file_name, edgeImg);
            //extract lines inside of the NMS area
            drawAnchorPoints(imgInfo->dirImg.ptr(), nmsImg.ptr(), anchors_filtered, edgeImg.ptr());
//            for (int j = 0; j < anchors_new.size(); ++j) {
//                cv::Point2f anchor(anchors_new[j].x, anchors_new[j].y);
//                cv::circle(edgeImg, anchor,3,255);
//            }
//            file_name = "/tmp/edgeImg_" + std::to_string(i) + ".jpg";
//            cv::imwrite(file_name, edgeImg);
//            std::cout << "save edge images, done\n";

            //record and clear segments
            nms_lines[i] = segments;

            //reset NMS area
//            std::cout << "refresh()\n";
            refresh();
            cv::fillConvexPoly(nmsImg, vertices, 4, UPM_ED_NO_EDGE_PIXEL);
//            std::cout << "fillRectangle again\n";
//            fillRectangle(nmsImg, vertices2f, 4, UPM_ED_NO_EDGE_PIXEL);
//            std::cout << "fillRectangle again, done\n";

            //todo reset edgeImg
//            edgeImg = cv::Mat::zeros(imgInfo->imageHeight, imgInfo->imageWidth, CV_8UC1);
        }
        if (cout_debug)
            std::cout << "extractLinesinsideNMS, done\n";
    }

    void ELSED::refresh()
    {
//        imgInfo = nullptr;
        edges.clear();
        segments.clear();
        salientSegments.clear();
//        anchors.clear();
//        drawer = nullptr;
//        blurredImg = cv::Mat();
//        edgeImg = cv::Mat();
//        nmsImg = cv::Mat();
        if (drawer)
            drawer->refresh();
    }

    void ELSED::filterAnchorsInsideNMSArea(const cv::Mat& preset_nmsImg, const std::vector<Pixel> &anchors_input, std::vector<Pixel> &inside_nms_anchors,
                                           std::vector<Pixel> &outside_nms_anchors)
    {
//        std::cout << "filterAnchorsInsideNMSArea\n";
//        cv::Mat nms_all;
//        nms_all = cv::Mat::zeros(imgInfo->imageHeight, imgInfo->imageWidth, CV_8UC1);
//        setLinesNMSArea(nms_all);

        int imageWidth = imgInfo->imageWidth;
        int imageHeight = imgInfo->imageHeight;
        int indexInArray;
        const auto  *pNMSImg = preset_nmsImg.ptr<uint8_t>();

        //filter anchor points
        int NMS_region_size = linesExist.size();
        outside_nms_anchors.resize(anchors.size());
        inside_nms_anchors.resize(anchors.size());
        int j = 0, k = 0;
        for (int i = 0; i < anchors.size(); ++i)
        {
            const Pixel& anchorPoint = anchors[i];
            indexInArray = anchorPoint.y * imageWidth + anchorPoint.x;
            if (pNMSImg[indexInArray])
                inside_nms_anchors[k++] = anchorPoint;
            else
                outside_nms_anchors[j++] = anchorPoint;
        }
        outside_nms_anchors.resize(j);
        inside_nms_anchors.resize(k);

//        std::cout << "filterAnchorsInsideNMSArea, done\n";
    }
}  // namespace upm
