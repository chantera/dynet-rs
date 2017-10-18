/* Copyright 2017 Hiroki Teranishi. All rights reserved. */

#include "dynetc/c_api.h"
#include "dynetc/c_api_internal.h"

#include "dynet/dynet.h"
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

CDim* CDim_new() {
  return new CDim;
}

CDim* CDim_new_from_array(const long* x) {
  return new CDim{std::vector<long>(x, x + sizeof x / sizeof x[0])};
}

void CDim_delete(CDim* d) {
  delete d;
}

int CDim_size(CDim* d) {
  return d->dim.size();
}

float C_as_scalar(const CTensor* t) {
  return dynet::as_scalar(t->tensor);
}

float* C_as_vector(const CTensor* v) {
  return &(dynet::as_vector(v->tensor))[0];
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
  return new CTensor{*(p->param.values())};
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
  return new CTensor{g->graph.forward(*CAST_TO_EXPR_PTR(last))};
}

void CComputationGraph_backward(CComputationGraph* g, const CExpression* last) {
  g->graph.backward(*CAST_TO_EXPR_PTR(last));
}

CExpression C_input_scalar(CComputationGraph* g, float s) {
  Expression expr = dynet::input(g->graph, s);
  return *CAST_TO_CEXPR_PTR(&expr);
}

CExpression C_input_vector(CComputationGraph* g, const CDim* d,
                           const float* data) {
  auto v = std::vector<float>(data, data + sizeof data / sizeof data[0]);
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

/*

CExpression C_op_add(const CExpression* x, const CExpression* y) {
  Expression expr = x->expr + y->expr;
  return *reinterpret_cast<CExpression*>(&expr);
}

CExpression C_op_mul(const CExpression* x, const CExpression* y) {
  Expression expr = x->expr * y->expr;
  return *reinterpret_cast<CExpression*>(&expr);
}

CExpression C_tanh(const CExpression* x) {
  Expression expr = dynet::tanh(x->expr);
  return *reinterpret_cast<CExpression*>(&expr);
}

CExpression C_squared_distance(const CExpression* x, const CExpression* y) {
  Expression expr = dynet::squared_distance(x->expr, y->expr);
  return *reinterpret_cast<CExpression*>(&expr);
}
 */

}  // end extern "C"
