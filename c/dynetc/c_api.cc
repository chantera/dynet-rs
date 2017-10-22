/* Copyright 2017 Hiroki Teranishi. All rights reserved. */

#include "dynetc/c_api.h"
#include "dynetc/c_api_internal.h"

#include "dynet/init.h"
#include "dynet/dim.h"
#include "dynet/tensor.h"
#include "dynet/model.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"

using dynet::ComputationGraph;
using dynet::Expression;
using dynet::Tensor;

#define CAST_TO_EXPR_PTR(x) reinterpret_cast<const Expression*>(x)

#define CAST_TO_CEXPR_PTR(x) reinterpret_cast<CExpression*>(x)

#define EXPR_UNARY_OP(name, expr_func) \
  CExpression name(const CExpression* x) { \
    auto x_expr = *CAST_TO_EXPR_PTR(x); \
    Expression expr = dynet::expr_func(x_expr); \
    return *CAST_TO_CEXPR_PTR(&expr); \
  }

#define EXPR_BINARY_OP(name, expr_func) \
  CExpression name(const CExpression* x, const CExpression* y) { \
    auto x_expr = *CAST_TO_EXPR_PTR(x); \
    auto y_expr = *CAST_TO_EXPR_PTR(y); \
    Expression expr = dynet::expr_func(x_expr, y_expr); \
    return *CAST_TO_CEXPR_PTR(&expr); \
  }

extern "C" {

CDynetParams* CDynetParams_new() {
  return new CDynetParams{};
}

void CDynetParams_delete(CDynetParams* p) {
  delete p;
}

void C_initialize_from_params(CDynetParams* params) {
  dynet::initialize(params->params);
}

void C_initialize(int argc, char* argv[]) {
  dynet::initialize(argc, argv);
}

CDim* CDim_new() {
  return new CDim;
}

CDim* CDim_new_from_array(const long* x, size_t n) {
  return new CDim{std::vector<long>(x, x + n)};
}

void CDim_delete(CDim* d) {
  delete d;
}

unsigned CDim_size(CDim* d) {
  return d->dim.size();
}

unsigned CDim_ndims(CDim* d) {
  return d->dim.ndims();
}

unsigned CDim_rows(CDim* d) {
  return d->dim.rows();
}

unsigned CDim_cols(CDim* d) {
  return d->dim.cols();
}

unsigned CDim_batch_elems(CDim* d) {
  return d->dim.batch_elems();
}

unsigned CDim_dim_size(CDim* d, unsigned i) {
  return d->dim.size(i);
}

}  // end extern "C"

namespace dynet {
namespace {

Tensor ctensor_to_tensor(const CTensor* src) {
  return {src->d.dim, src->v, src->device, src->mem_pool};
}

CTensor* ctensor_from_tensor(const Tensor& src) {
  return new CTensor{{src.d}, src.v, src.device, src.mem_pool};
}

}  // namespace
}  // namespace dynet

extern "C" {

void CTensor_delete(CTensor* t) {
  delete t;
}

CDim* CTensor_dim(CTensor* t) {
  return &(t->d);
}

float C_as_scalar(const CTensor* t) {
  return dynet::as_scalar(dynet::ctensor_to_tensor(t));
}

float* C_as_vector(const CTensor* v) {
  return &(dynet::as_vector(dynet::ctensor_to_tensor(v)))[0];
}

CParameter* CParameter_new() {
  return new CParameter;
}

void CParameter_delete(CParameter* p) {
  delete p;
}

void CParameter_zero(CParameter* p) {
  p->param.zero();
}

CDim* CParameter_dim(CParameter* p) {
  return new CDim{p->param.dim()};
}

CTensor* CParameter_values(CParameter* p) {
  return dynet::ctensor_from_tensor(*(p->param.values()));
}

void CParameter_set_updated(CParameter* p, bool b) {
  p->param.set_updated(b);
}

bool CParameter_is_updated(CParameter* p) {
  return p->param.is_updated();
}

CLookupParameter* CLookupParameter_new() {
  return new CLookupParameter;
}

void CLookupParameter_delete(CLookupParameter* p) {
  delete p;
}

void CLookupParameter_zero(CLookupParameter* p) {
  p->param.zero();
}

CDim* CLookupParameter_dim(CLookupParameter* p) {
  return new CDim{p->param.dim()};
}

// CTensor* CLookupParameter_values(CLookupParameter* p) {
//   return new CTensor{*(p->param.values())};
// }

void CLookupParameter_set_updated(CLookupParameter* p, bool b) {
  p->param.set_updated(b);
}

bool CLookupParameter_is_updated(CLookupParameter* p) {
  return p->param.is_updated();
}

CParameterCollection* CParameterCollection_new() {
  return new CParameterCollection;
}

void CParameterCollection_delete(CParameterCollection* pc) {
  delete pc;
}

float CParameterCollection_gradient_l2_norm(CParameterCollection* pc) {
  return pc->pc.gradient_l2_norm();
}

void CParameterCollection_reset_gradient(CParameterCollection* pc) {
  pc->pc.reset_gradient();
}

CParameter* CParameterCollection_add_parameters(CParameterCollection* pc,
                                               const CDim* d) {
  return new CParameter{pc->pc.add_parameters(d->dim)};
}

CLookupParameter* CParameterCollection_add_lookup_parameters(
    CParameterCollection* pc, unsigned n, const CDim* d) {
  return new CLookupParameter{pc->pc.add_lookup_parameters(n, d->dim)};
}

CComputationGraph* CComputationGraph_new() {
  return new CComputationGraph;
}

void CComputationGraph_delete(CComputationGraph* g) {
  delete g;
}

const CTensor* CComputationGraph_forward(CComputationGraph* g,
                                         const CExpression* last) {
  return dynet::ctensor_from_tensor(g->graph.forward(*CAST_TO_EXPR_PTR(last)));
}

void CComputationGraph_backward(CComputationGraph* g, const CExpression* last) {
  g->graph.backward(*CAST_TO_EXPR_PTR(last));
}

void CTrainer_update(void* t) {
  reinterpret_cast<dynet::Trainer*>(t)->update();
}

CSimpleSGDTrainer* CSimpleSGDTrainer_new(CParameterCollection* m,
                                         float learning_rate) {
  return new CSimpleSGDTrainer{dynet::SimpleSGDTrainer{m->pc, learning_rate}};
}

void CSimpleSGDTrainer_delete(CSimpleSGDTrainer* t) {
  delete t;
}

CExpression C_input_scalar(CComputationGraph* g, float s) {
  Expression expr = dynet::input(g->graph, s);
  return *CAST_TO_CEXPR_PTR(&expr);
}

CExpression C_input_vector(CComputationGraph* g, const CDim* d,
                           const float* data, size_t n) {
  auto v = std::vector<float>(data, data + n);
  Expression expr = dynet::input(g->graph, d->dim, v);
  return *CAST_TO_CEXPR_PTR(&expr);
}

CExpression C_parameter(CComputationGraph* g, CParameter* p) {
  Expression expr = dynet::parameter(g->graph, p->param);
  return *CAST_TO_CEXPR_PTR(&expr);
}

CExpression C_lookup_parameter(CComputationGraph* g, CLookupParameter* p) {
  Expression expr = dynet::parameter(g->graph, p->param);
  return *CAST_TO_CEXPR_PTR(&expr);
}

EXPR_BINARY_OP(C_op_add, operator+)
EXPR_BINARY_OP(C_op_mul, operator*)
EXPR_UNARY_OP(C_tanh, tanh)
EXPR_BINARY_OP(C_squared_distance, squared_distance)

}  // end extern "C"
