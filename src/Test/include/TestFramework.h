#ifndef __TEST_FRAMEWORK_H__
#define __TEST_FRAMEWORK_H__

#include "MattingPlugin.h"

class TestFramework {
 public:
  TestFramework();
  virtual ~TestFramework();

 public:
  void run(int argc, char* argv[]);

 private:
  MattingPlugin* m_mattingPlugin;
};
#endif