#ifndef ELSED_ELSED_H_
#define ELSED_ELSED_H_

#include <ostream>

#include "FullSegmentInfo.h"
#include "EdgeDrawer.h"

namespace upm {

struct ELSEDParams {
  // Gaussian kernel size
  int ksize = 5;
  // Sigma of the gaussian kernel
  float sigma = 1;
  // The threshold of pixel gradient magnitude.
  // Only those pixels whose gradient magnitude are larger than
  // this threshold will be taken as possible edge points.
  float gradientThreshold = 40;
  // If the pixel's gradient value is bigger than both of its neighbors by a
  // certain anchorThreshold, the pixel is marked to be an anchor.
  uint8_t anchorThreshold = 10; //todo 8
  // Anchor testing can be performed at different scan intervals, i.e.,
  // every row/column, every second row/column
  unsigned int scanIntervals = 9;//todo

  // Minimum line segment length
  int minLineLen = 50;
  // Threshold used to check if a list of edge points for a line segment
  double lineFitErrThreshold = 0.2;
  // Threshold used to check if a new pixel is part of an already fit line segment
  double pxToSegmentDistTh = 1.5;
  // Threshold used to validate the junction jump region. The first eigenvalue of the gradient
  // auto-correlation matrix should be at least junctionEigenvalsTh times bigger than the second eigenvalue
  double junctionEigenvalsTh = 10;
  // the difference between the perpendicular segment direction and the direction of the gradient
  // in the region to be validated must be less than junctionAngleTh radians
  double junctionAngleTh = 10 * (M_PI / 180.0);
  // The threshold over the validation criteria. For ELSED, it is the gradient angular error in pixels.
  double validationTh = 0.15;

  // Whether to validate or not the generated segments
  bool validate = true;
  // Whether to jump over junctions
  bool treatJunctions = true;
  // List of junction size that will be tested (in pixels)
  std::vector<int> listJunctionSizes = {9, 11, 13};

  //todo NMS长宽大小
  //NMS extend length
  int nms_height_extend = 30;
  int nms_width_extend = 100;
};

struct lineInfo{
    cv::Point2f centerPoint, startPoint, endPoint;
    float length;
    float angle; // degree
    cv::Point2f unitDir;
    bool need_detect;
    std::vector<cv::Point2f> sample_points;
};

/**
 * This class implements the method:
 *     @cite Suárez, I., Buenaposada, J. M., & Baumela, L. (2021).
 *     ELSED: Enhanced Line SEgment Drawing. arXiv preprint arXiv:2108.03144.
 *
 * It is an efficient line segment detector amenable to use in low power devices such as drones or smartphones.
 * The method takes an image as input and outputs a list of detected segments.
 */
class ELSED {
 public:
  // Constructor
  explicit ELSED(const ELSEDParams &params = ELSEDParams());

  /**
   * @brief Detects segments in the input image
   * @param image An input image. The parameters are adapted to images of size 640x480.
   * Bigger images will generate more segments.
   * @return The list of detected segments
   */
  Segments detect(const cv::Mat &image);

  SalientSegments detectSalient(const cv::Mat &image);

  ImageEdges detectEdges(const cv::Mat &image);  // NOLINT

  const LineDetectionExtraInfo &getImgInfo() const;

  const LineDetectionExtraInfoPtr &getImgInfoPtr() const;

  void processImage(const cv::Mat &image);

  void clear();

  static void computeAnchorPoints(const cv::Mat &dirImage,
                                  const cv::Mat &gradImageWO,
                                  const cv::Mat &gradImage,
                                  const cv::Mat &nmsImage,
                                  int scanInterval,
                                  int anchorThresh,
                                  std::vector<Pixel> &anchorPoints);  // NOLINT

  static LineDetectionExtraInfoPtr
  computeGradients(const cv::Mat &srcImg, short gradientTh);

  ImageEdges getAllEdges() const;

  ImageEdges getSegmentEdges() const;

  const EdgeDrawerPtr &getDrawer() const { return drawer; }

  void DrawRotatedRectangle(cv::Mat& image, const cv::Point2f& centerPoint, const cv::Size& rectangleSize,
                            const float& rotationDegrees, const int& val = UPM_ED_NMS_PX);

  void setLinesExist(const Segments& linesExist);

  std::vector<lineInfo>& getLinesExist() { return linesExist; }

  //from predictde lines
  void setLinesNMSArea();
  void setLinesNMSArea(cv::Mat& image);
  void setAnchorsExist();

  void filterAnchorsInsideNMSRegion(std::vector<Pixel> &anchorPoints);
  void filterAnchorsInsideNMSArea(const std::vector<Pixel> &anchors_input, std::vector<std::vector<Pixel>> &inside_nms_anchors,
                                  std::vector<Pixel> &outside_nms_anchors);
  void filterAnchorsInsideNMSArea(const cv::Mat& preset_nmsImg, const std::vector<Pixel> &anchors_input, std::vector<Pixel> &inside_nms_anchors,
                                  std::vector<Pixel> &outside_nms_anchors);

  Segments& getELSEDSegments() {return segments;}

  std::vector<Segments>& getNMSSegments() {return nms_lines;}

  void setNMSExtend(const int& nms_extend) {params.nms_height_extend = nms_extend;}

  void extractLinesinsideNMS();

  static bool compareLength(Segment& s1, Segment& s2)
  {
      return (s1(0) - s1(2)) * (s1(0) - s1(2)) + (s1(1) - s1(3)) * (s1(1) - s1(3)) >
              (s2(0) - s2(2)) * (s2(0) - s2(2)) + (s2(1) - s2(3)) * (s2(1) - s2(3));
  }

  void refresh();

  Segments outside_nms_lines;
  bool enough_lines = false;


private:
  void drawAnchorPoints(const uint8_t *dirImg,
                        const uint8_t *pNMSImg,
                        const std::vector<Pixel> &anchorPoints,
                        uint8_t *pEdgeImg);  // NOLINT

  void fillRectangle(cv::Mat& img, const cv::Point2f* pts, const int& npts, const uint8_t& color);

  void addAnchors(const cv::Mat &dirImage,
                  const cv::Mat &gradImageWO,
                  const cv::Mat &gradImage,
                  const cv::Mat &nmsImage,
                  int scanInterval,
                  int anchorThresh,
                  std::vector<cv::Point2f> candidates,
                  std::vector<Pixel> &anchorPoints);

  ELSEDParams params;
  LineDetectionExtraInfoPtr imgInfo;
  ImageEdges edges;
  Segments segments;
  std::vector<int> full_segments_id;//valid segments in drawer->getDetectedFullSegments()
  SalientSegments salientSegments;
  std::vector<Pixel> anchors;
  EdgeDrawerPtr drawer;
  EdgeDrawerPtr drawer_tmp;
  cv::Mat blurredImg;
  cv::Mat edgeImg;
  cv::Mat nmsImg; //from predict lines

  std::vector<Pixel> anchorsExist;//from predict lines

  std::vector<Segments> nms_lines; // extract from certain NMS region

  std::vector<Pixel> inside_nms_anchors;

  bool one_line_inside_NMS_area = false;

  struct length_id
  {
      float length;
      int segments_id;
      int global_segments_id;
  };

  //For non maximum suppression
  std::vector<lineInfo> linesExist;
};
}
#endif //ELSED_ELSED_H_
