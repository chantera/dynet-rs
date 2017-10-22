/* Copyright 2017 Hiroki Teranishi. All rights reserved. */

#ifndef DYNETC_C_API_INTERNAL_H_
#define DYNETC_C_API_INTERNAL_H_

#include "dynetc/c_api.h"

#include "dynet/init.h"
#include "dynet/dim.h"
#include "dynet/tensor.h"
#include "dynet/model.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"

struct CDynetParams {
  dynet::DynetParams params;
};

struct CDim {
  dynet::Dim dim;
};

struct CTensor {
  CDim d;
  float* v;
  dynet::Device* device;
  dynet::DeviceMempool mem_pool;
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

struct CSimpleSGDTrainer {
  dynet::SimpleSGDTrainer trainer;
};

#endif  // DYNETC_C_API_INTERNAL_H_