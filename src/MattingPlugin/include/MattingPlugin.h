#ifndef __MATTING_PLUGIN_H__
#define __MATTING_PLUGIN_H__

#include <torch/script.h>

#include <cstring>
#include <opencv2/opencv.hpp>

#include "SysDef.h"

class MattingPlugin {
 public:
  MattingPlugin();
  virtual ~MattingPlugin();

 public:
  bool LoadModel(std::string path);
  cv::Mat ReadImg(std::string path);
  cv::Mat Matting(const char* path);
  cv::Mat Matting(cv::Mat img);
  void SaveImg(std::string path, cv::Mat img);

 protected:
  torch::Tensor Mat2Tensor(cv::Mat mat);
  cv::Mat Tensor2Mat(torch::Tensor tensor);
  torch::Tensor Pretreatment(torch::Tensor& tensor);
  auto MattingInference(torch::Tensor& tensor);
  torch::Tensor PostProcessing(torch::Tensor& tensor);
  torch::Tensor GeenBackGround(torch::Tensor fgr, torch::Tensor pha);

 private:
  torch::Tensor HWC2CHW(torch::Tensor& tensor);
  torch::Tensor CHW2HWC(torch::Tensor& tensor);
  torch::Tensor AddBatchSizeDim(torch::Tensor& tensor);
  torch::Tensor RemoveBatchSizeDim(torch::Tensor& tensor);

 private:
  torch::jit::script::Module m_model;
};
#endif