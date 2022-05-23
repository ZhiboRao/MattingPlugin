#include "TestFramework.h"

#include <iostream>

TestFramework::TestFramework() : m_mattingPlugin(nullptr) {}
TestFramework::~TestFramework() {}

void TestFramework::run(int argc, char* argv[]) {
  std::cout << "hello world!" << std::endl;
  m_mattingPlugin = new MattingPlugin();
  m_mattingPlugin->LoadModel("./checkpoint/model_matting_jit.pt");
  auto img = m_mattingPlugin->Matting("./example/example.jpeg");
  m_mattingPlugin->SaveImg("./example/1.png", img);
  std::cout << "hello world 1!" << std::endl;
  delete m_mattingPlugin;
  std::cout << "hello world 2!" << std::endl;
  m_mattingPlugin = nullptr;
}