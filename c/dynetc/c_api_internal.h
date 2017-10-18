/* Copyright 2017 Hiroki Teranishi. All rights reserved. */

#ifndef DYNETC_C_API_INTERNAL_H_
#define DYNETC_C_API_INTERNAL_H_

#include "dynetc/c_api.h"

#include "dynet/dynet.h"
#include "dynet/dim.h"
#include "dynet/expr.h"
#include "dynet/model.h"
#include "dynet/tensor.h"

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

#endif  // DYNETC_C_API_INTERNAL_H_