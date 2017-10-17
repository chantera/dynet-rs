#ifndef DYNET_C_C_API_INTERNAL_H_
#define DYNET_C_C_API_INTERNAL_H_

#include "c_api.h"

#include "dynet/dynet.h"
#include "dynet/dim.h"
#include <dynet/expr.h>
#include <dynet/model.h>
#include "dynet/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

struct CDynetParams {
  dynet::DynetParams params;
};

struct CDim {
  dynet::Dim dim;
};

struct CTensor {
  dynet::Tensor tensor;
};

struct CParameter {
  dynet::Parameter param;
};

struct CLookupParameter {
  dynet::LookupParameter param;
};

struct CParameterCollection {
  dynet::ParameterCollection pc;
};

struct CComputationGraph {
  dynet::ComputationGraph graph;
};

struct CExpression {
  dynet::Expression expr;
};


#ifdef __cplusplus
} /* end extern "C" */
#endif

void hello();
#endif  // DYNET_C_C_API_INTERNAL_H_