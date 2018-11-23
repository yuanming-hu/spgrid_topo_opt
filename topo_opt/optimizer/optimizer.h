#include <taichi/common/util.h>
#include "../fem_interface.h"

TC_NAMESPACE_BEGIN

class Optimizer : public Unit {
 public:
  Dict config;

  TC_IO_DECL {
  }

  Optimizer() : Unit() {
  }

  Optimizer(const Dict &config) : Unit() {
    this->config = config;
  }

  // Returns max density change
  virtual real optimize(const Dict &param) {
    TC_NOT_IMPLEMENTED;
    return 0.0_f;
  }

  virtual std::string get_name() const override {
    return "topOpt_optimizer";
  }

};

TC_INTERFACE(Optimizer);

TC_NAMESPACE_END

