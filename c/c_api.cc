#include "c_api.h"
#include "c_api_internal.h"

#include "dynet/dynet.h"
#include "dynet/expr.h"

using dynet::ComputationGraph;
using dynet::Expression;
using dynet::Tensor;

extern "C" {

CDim* CDim_new() {
  return new CDim;
}

CDim* CDim_new_from_array(const long* x) {
  return new CDim(std::vector<long>(x, x + sizeof x / sizeof x[0]));
}

void CDim_delete(CDim* d) {
  delete d;
}

int CDim_size(CDim* d) {
  return d->dim.size();
}

real C_as_scalar(const CTensor* t) {
  return dynet::as_scalar(t->tensor);
}

real* C_as_vector(const CTensor* v) {
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

CDim CParameter_dim(CParameter* p) {
  return {p->param.dim()};
}

CTensor* CParameter_values(CParameter* p) {
  return new CTensor(p->param.values());
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
  return new CDim(p->param.dim());
}

CTensor* CLookupParameter_values(CLookupParameter* p) {
  return new CTensor(p->param.values());
}

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
  return new CParameter(pc->pc.add_parameters(d->dim));
}

CLookupParameter* CParameterCollection_add_lookup_parameters(
    CParameterCollection* pc, unsigned n, const CDim* d) {
  return new CLookupParameter(pc->pc.add_lookup_parameters(n, d->dim));
}

CComputationGraph* CComputationGraph_new() {
  return new CComputationGraph;
}

void CComputationGraph_delete(CComputationGraph* g) {
  delete g;
}

const CTensor* CComputationGraph_forward(CComputationGraph* g,
                                         const CExpression* last) {
  const Tensor tensor = g->graph.forward(last->expr);
  return new CTensor(tensor);
}

void CComputationGraph_backward(CComputationGraph* g, const CExpression* last) {
  g->graph.backward(last->expr);
}

CExpression* C_input_scalar(CComputationGraph* g, real s) {
  return new CExpression(dynet::input(g->graph, s));
}

CExpression* C_input_vector(CComputationGraph* g, const CDim* d,
                            const float* data) {
  auto v = std::vector<float>(data, data + sizeof data / sizeof data[0]);
  return new CExpression(dynet::input(g->graph, d->dim, v));
}

CExpression* C_parameter(CComputationGraph* g, CParameter* p) {
  return new CExpression(dynet::parameter(g->graph, p->param));
}
CExpression* C_lookup_parameter(CComputationGraph* g, CLookupParameter* p) {
  return new CExpression(dynet::parameter(g->graph, p->param));
}

CExpression* C_op_add(const CExpression* x, const CExpression* y) {
  return new CExpression(x->expr + y->expr);
}

CExpression* C_op_mul(const CExpression* x, const CExpression* y) {
  return new CExpression(x->expr * y->expr);
}

CExpression* C_tanh(const CExpression* x) {
  return new CExpression(dynet::tanh(x->expr));
}

CExpression* C_squared_distance(const CExpression* x, const CExpression* y) {
  return new CExpression(dynet::squared_distance(x->expr, y->expr));
}

}  // end extern "C"
