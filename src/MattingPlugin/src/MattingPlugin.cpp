#include "MattingPlugin.h"

#include <cassert>
#include <iostream>

MattingPlugin::MattingPlugin(){};
MattingPlugin::~MattingPlugin(){};

cv::Mat Tensor2Mat1(torch::Tensor tensor) {
  tensor = tensor.mul(RGB_NUM).clamp(0, RGB_NUM).to(torch::kU8);
  cv::Mat cv_mat(tensor.sizes()[0], tensor.sizes()[1], CV_8UC1);
  std::memcpy((void*)cv_mat.data, tensor.data_ptr(),
              sizeof(torch::kU8) * tensor.numel());
  return cv_mat;
}

bool MattingPlugin::LoadModel(std::string path) {
  try {
    m_model = torch::jit::load(path);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return false;
  }
  return true;
}

cv::Mat MattingPlugin::ReadImg(std::string path) {
  std::cout << path << std::endl;
  return cv::imread(path, cv::IMREAD_COLOR);
}

cv::Mat MattingPlugin::Matting(const char* path) {
  return Matting(ReadImg(path));
}

torch::Tensor MattingPlugin::Mat2Tensor(cv::Mat mat) {
  return torch::from_blob(mat.data, {mat.rows, mat.cols, RGB_CHANNEL_NUM},
                          at::kByte);
}

cv::Mat MattingPlugin::Tensor2Mat(torch::Tensor tensor) {
  tensor = tensor.mul(RGB_NUM).clamp(0, RGB_NUM).to(torch::kU8);
  cv::Mat cv_mat(tensor.sizes()[0], tensor.sizes()[1], CV_8UC3);
  std::memcpy((void*)cv_mat.data, tensor.data_ptr(),
              sizeof(torch::kU8) * tensor.numel());
  return cv_mat;
}

torch::Tensor MattingPlugin::HWC2CHW(torch::Tensor& tensor) {
  return tensor.permute({2, 0, 1});
}

torch::Tensor MattingPlugin::CHW2HWC(torch::Tensor& tensor) {
  return tensor.permute({1, 2, 0});
}

torch::Tensor MattingPlugin::AddBatchSizeDim(torch::Tensor& tensor) {
  return torch::unsqueeze(tensor, 0);
}

torch::Tensor MattingPlugin::RemoveBatchSizeDim(torch::Tensor& tensor) {
  return torch::squeeze(tensor, 0);
}

torch::Tensor MattingPlugin::Pretreatment(torch::Tensor& tensor) {
  tensor = HWC2CHW(tensor);
  tensor = AddBatchSizeDim(tensor);
  tensor = tensor / RGB_NUM;
  return tensor;
}

auto MattingPlugin::MattingInference(torch::Tensor& tensor) {
  std::vector<torch::IValue> inputs;
  inputs.push_back(tensor);
  return m_model.forward(inputs).toTuple();
}

torch::Tensor MattingPlugin::GeenBackGround(torch::Tensor fgr,
                                            torch::Tensor pha) {
  torch::Tensor bgr =
      torch::tensor({BACKGROUND_R, BACKGROUND_G, BACKGROUND_B}).view({3, 1, 1});
  return fgr * pha + bgr * (1 - pha);
}

torch::Tensor MattingPlugin::PostProcessing(torch::Tensor& tensor) {
  tensor = RemoveBatchSizeDim(tensor);
  tensor = CHW2HWC(tensor);
  return tensor;
}

cv::Mat MattingPlugin::Matting(cv::Mat img) {
  torch::Tensor tensor_img = Mat2Tensor(img);
  tensor_img = Pretreatment(tensor_img);
  auto outputs = MattingInference(tensor_img);
  tensor_img = GeenBackGround(tensor_img, outputs->elements()[1].toTensor());
  std::cout << tensor_img.sizes() << std::endl;
  tensor_img = PostProcessing(tensor_img);
  std::cout << tensor_img.sizes() << std::endl;
  img = Tensor2Mat(tensor_img);
  return img;
}

void MattingPlugin::SaveImg(std::string path, cv::Mat img) {
  cv::imwrite(path, img);
}
