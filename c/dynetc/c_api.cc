/* Copyright 2017 Hiroki Teranishi. All rights reserved. */

#include "dynetc/c_api.h"

#include <vector>

#include "dynetc/c_api_internal.h"
#include "dynet/init.h"
#include "dynet/dim.h"
#include "dynet/tensor.h"
#include "dynet/model.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/rnn.h"
#include "dynet/lstm.h"

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

CExpression C_lookup(CComputationGraph* g,
                     CLookupParameter* p, unsigned index) {
  Expression expr = dynet::lookup(g->graph, p->param, index);
  return *CAST_TO_CEXPR_PTR(&expr);
}

CExpression C_lookup_batch(CComputationGraph* g,
                           CLookupParameter* p, const unsigned* indices) {
  Expression expr = dynet::lookup(g->graph, p->param, indices);
  return *CAST_TO_CEXPR_PTR(&expr);
}

CExpression C_const_lookup(CComputationGraph* g,
                           CLookupParameter* p, unsigned index) {
  Expression expr = dynet::const_lookup(g->graph, p->param, index);
  return *CAST_TO_CEXPR_PTR(&expr);
}

CExpression C_const_lookup_batch(CComputationGraph* g,
                                 CLookupParameter* p, const unsigned* indices) {
  Expression expr = dynet::const_lookup(g->graph, p->param, indices);
  return *CAST_TO_CEXPR_PTR(&expr);
}

EXPR_BINARY_OP(C_op_add, operator+)
EXPR_BINARY_OP(C_op_mul, operator*)
EXPR_UNARY_OP(C_tanh, tanh)
EXPR_BINARY_OP(C_squared_distance, squared_distance)

#include <iostream>
#include "dynet/nodes-concat.h"

CExpression C_concatenate(const CExpression* const xs[], size_t n, unsigned d) {
  std::vector<Expression> v;
  for (int i = 0; i < n; ++i) {
    auto e = *CAST_TO_EXPR_PTR(xs[i]);
    v.push_back(e);
  }
  // std::cout << "size: " << v.size() << std::endl;
  // for (Expression e : v) {
  //   std::cout << e.pg << std::endl;
  //   std::cout << e.i << std::endl;
  //   std::cout << e.graph_id << std::endl;
  //   std::cout << n << std::endl;
  //   std::cout << e.dim() << std::endl;
  //   std::cout << e.dim().batch_elems() << std::endl;
  //   std::cout << e.pg->nodes.size() << std::endl;
  // }
  Expression expr = dynet::concatenate(v, d);
  // Expression expr;
  // try {
  //   // expr = dynet::detail::f<dynet::Concatenate>(v, d);
  //   // expr = dynet::tanh(v[0]);
  //   // std::vector<dynet::real> x1_values{0, 1};
  //   // std::vector<dynet::real> x2_values{2, 3};
  //   std::cout << "====" << std::endl;
  //   std::cout << v[0].pg << std::endl;
  //   std::cout << v[0].i << std::endl;
  //   std::cout << v[0].graph_id << std::endl;
  //   std::cout << v[1].pg << std::endl;
  //   std::cout << v[1].i << std::endl;
  //   std::cout << v[1].graph_id << std::endl;
  //   expr = dynet::concatenate({v[0], v[1]}, 0);
  //   // {dynet::input(*v[0].pg, {2}, &x1_values),
  //   // dynet::input(*v[0].pg, {2}, &x2_values)}, 0);
  //   std::cout << "====" << std::endl;
  // } catch (std::exception &e) {
  //   std::cerr << e.what() << std::endl;
  // }
  return *CAST_TO_CEXPR_PTR(&expr);
}

void CRNNBuilder_start_new_sequence_with_initial_hidden_states(
    void* builder, const CExpression* const h_0[], size_t n) {
  auto v = std::vector<Expression>(n);
  for (int i = 0; i < n; ++i) {
    v.push_back(*CAST_TO_EXPR_PTR(h_0[i]));
  }
  reinterpret_cast<dynet::RNNBuilder*>(builder)->start_new_sequence(v);
}

void CRNNBuilder_new_graph(void* builder, CComputationGraph* cg, bool update) {
  reinterpret_cast<dynet::RNNBuilder*>(builder)->new_graph(cg->graph, update);
}

CExpression CRNNBuilder_add_input(void* builder, const CExpression* x) {
  auto x_expr = *CAST_TO_EXPR_PTR(x); \
  Expression expr =
      reinterpret_cast<dynet::RNNBuilder*>(builder)->add_input(x_expr);
  return *CAST_TO_CEXPR_PTR(&expr);
}

CSimpleRNNBuilder* CSimpleRNNBuilder_new(unsigned layers,
                                        unsigned input_dim,
                                        unsigned hidden_dim,
                                        CParameterCollection* model,
                                        bool support_lags) {
  return new CSimpleRNNBuilder{
      dynet::SimpleRNNBuilder{
          layers, input_dim, hidden_dim, model->pc, support_lags}};
}

void CSimpleRNNBuilder_delete(CSimpleRNNBuilder* builder) {
  delete builder;
}

void CSimpleRNNBuilder_start_new_sequence(CSimpleRNNBuilder* builder) {
  builder->builder.start_new_sequence();
}

void CSimpleRNNBuilder_start_new_sequence_with_initial_hidden_states(
    CSimpleRNNBuilder* builder, const CExpression* const h_0[], size_t n) {
  auto v = std::vector<Expression>(n);
  for (int i = 0; i < n; ++i) {
    v.push_back(*CAST_TO_EXPR_PTR(h_0[i]));
  }
  builder->builder.start_new_sequence(v);
}


// ---------------- declarations from lstm.h ----------------

/**
 * SimpleRNNBuilder
 */
CVanillaLSTMBuilder* CVanillaLSTMBuilder_new(unsigned layers,
                                             unsigned input_dim,
                                             unsigned hidden_dim,
                                             CParameterCollection* model,
                                             bool ln_lstm) {
  return new CVanillaLSTMBuilder{
      dynet::VanillaLSTMBuilder{
          layers, input_dim, hidden_dim, model->pc, ln_lstm}};
}

void CVanillaLSTMBuilder_delete(CVanillaLSTMBuilder* builder) {
  delete builder;
}

void CVanillaLSTMBuilder_start_new_sequence(CVanillaLSTMBuilder* builder) {
  builder->builder.start_new_sequence();
}

void CVanillaLSTMBuilder_start_new_sequence_with_initial_hidden_states(
    CVanillaLSTMBuilder* builder, const CExpression* const h_0[], size_t n) {
  auto v = std::vector<Expression>(n);
  for (int i = 0; i < n; ++i) {
    v.push_back(*CAST_TO_EXPR_PTR(h_0[i]));
  }
  builder->builder.start_new_sequence(v);
}

void CVanillaLSTMBuilder_set_dropout(CVanillaLSTMBuilder* builder,
                                     float d,
                                     float d_r) {
  builder->builder.set_dropout(d, d_r);
}

void CVanillaLSTMBuilder_disable_dropout(CVanillaLSTMBuilder* builder) {
  builder->builder.disable_dropout();
}

}  // end extern "C"
