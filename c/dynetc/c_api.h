/* Copyright 2017 Hiroki Teranishi. All rights reserved. */

#ifndef DYNETC_C_API_H_
#define DYNETC_C_API_H_

#include <stddef.h>  // NOLINT
#include <stdint.h>  // NOLINT

#ifndef __bool_true_false_are_defined
#define __bool_true_false_are_defined 1
#ifndef __cplusplus

#ifndef _Bool
#define _Bool unsigned char
#endif

typedef _Bool bool;
#define true  1
#define false 0

#endif /* __cplusplus */
#endif /* __bool_true_false_are_defined */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CDynetParams CDynetParams;

typedef struct CDim CDim;

typedef struct CTensor CTensor;

typedef struct CParameter CParameter;

typedef struct CLookupParameter CLookupParameter;

typedef struct CParameterCollection CParameterCollection;

typedef struct CComputationGraph CComputationGraph;

typedef struct CSimpleSGDTrainer CSimpleSGDTrainer;

typedef struct CExpression {
  void* cg;
  unsigned v_index;
  unsigned graph_id;
} CExpression;

typedef struct CSimpleRNNBuilder CSimpleRNNBuilder;

typedef struct CVanillaLSTMBuilder CVanillaLSTMBuilder;


// ---------------- declarations from init.h ----------------

/**
 * DynetParams
 */
CDynetParams* CDynetParams_new();

void CDynetParams_delete(CDynetParams* p);

/**
 * functions
 */
void C_initialize_from_params(CDynetParams* params);

void C_initialize(int argc, char* argv[]);


// ---------------- declarations from dim.h ----------------

/**
 * Dim
 */
CDim* CDim_new();

CDim* CDim_new_from_array(const long* x, size_t n);

void CDim_delete(CDim* d);

unsigned CDim_size(CDim* d);

unsigned CDim_ndims(CDim* d);

unsigned CDim_rows(CDim* d);

unsigned CDim_cols(CDim* d);

unsigned CDim_batch_elems(CDim* d);

unsigned CDim_dim_size(CDim* d, unsigned i);


// ---------------- declarations from tensor.h ----------------

/**
 * Tensor
 */
void CTensor_delete(CTensor* t);

CDim* CTensor_dim(CTensor* t);

/**
 * functions
 */
float C_as_scalar(const CTensor* t);

float* C_as_vector(const CTensor* v);


// ---------------- declarations from model.h ----------------

/**
 * Parameter
 */
CParameter* CParameter_new();

void CParameter_delete(CParameter* p);

void CParameter_zero(CParameter* p);

CDim* CParameter_dim(CParameter* p);

CTensor* CParameter_values(CParameter* p);

void CParameter_set_updated(CParameter* p, bool b);

bool CParameter_is_updated(CParameter* p);

/**
 * LookupParameter
 */
CLookupParameter* CLookupParameter_new();

void CLookupParameter_delete(CLookupParameter* p);

void CLookupParameter_zero(CLookupParameter* p);

CDim* CLookupParameter_dim(CLookupParameter* p);

// CTensor* CLookupParameter_values(CLookupParameter* p);

void CLookupParameter_set_updated(CLookupParameter* p, bool b);

bool CLookupParameter_is_updated(CLookupParameter* p);

/**
 * ParameterCollection
 */
CParameterCollection* CParameterCollection_new();

void CParameterCollection_delete(CParameterCollection* pc);

float CParameterCollection_gradient_l2_norm(CParameterCollection* pc);

void CParameterCollection_reset_gradient(CParameterCollection* pc);

CParameter* CParameterCollection_add_parameters(CParameterCollection* pc,
                                               const CDim* d);

CLookupParameter* CParameterCollection_add_lookup_parameters(
    CParameterCollection* pc, unsigned n, const CDim* d);


// ---------------- declarations from io.h ----------------


// ---------------- declarations from param-init.h ----------------


// ---------------- declarations from dynet.h ----------------

/**
 * ComputationGraph
 */
CComputationGraph* CComputationGraph_new();

void CComputationGraph_delete(CComputationGraph* g);

const CTensor* CComputationGraph_forward(CComputationGraph* g,
                                         const CExpression* last);

void CComputationGraph_backward(CComputationGraph* g, const CExpression* last);


// ---------------- declarations from training.h ----------------

/**
 * Trainer
 */
void CTrainer_update(void* t);

/**
 * SimpleSGDTrainer
 */
CSimpleSGDTrainer* CSimpleSGDTrainer_new(CParameterCollection* m,
                                         float learning_rate);

void CSimpleSGDTrainer_delete(CSimpleSGDTrainer* t);


// ---------------- declarations from expr.h ----------------

CExpression C_input_scalar(CComputationGraph* g, float s);
CExpression C_input_vector(CComputationGraph* g, const CDim* d,
                            const float* data, size_t n);
CExpression C_parameter(CComputationGraph* g, CParameter* p);
CExpression C_lookup_parameter(CComputationGraph* g, CLookupParameter* p);
CExpression C_lookup(CComputationGraph* g, CLookupParameter* p, unsigned index);
CExpression C_lookup_batch(CComputationGraph* g,
                           CLookupParameter* p, const unsigned* indices);
CExpression C_const_lookup(CComputationGraph* g,
                           CLookupParameter* p, unsigned index);
CExpression C_const_lookup_batch(CComputationGraph* g,
                                 CLookupParameter* p, const unsigned* indices);

CExpression C_op_add(const CExpression* x, const CExpression* y);
CExpression C_op_mul(const CExpression* x, const CExpression* y);

CExpression C_tanh(const CExpression* x);

CExpression C_squared_distance(const CExpression* x, const CExpression* y);

CExpression C_concatenate(const CExpression* const xs[], size_t n, unsigned d);


// ---------------- declarations from rnn.h ----------------

/**
 * RNNBuilder
 */
void CRNNBuilder_new_graph(void* builder, CComputationGraph* cg, bool update);

CExpression CRNNBuilder_add_input(void* builder, const CExpression* x);

/**
 * SimpleRNNBuilder
 */
CSimpleRNNBuilder* CSimpleRNNBuilder_new(unsigned layers,
                                         unsigned input_dim,
                                         unsigned hidden_dim,
                                         CParameterCollection* model,
                                         bool support_lags);

void CSimpleRNNBuilder_delete(CSimpleRNNBuilder* builder);

void CSimpleRNNBuilder_start_new_sequence(CSimpleRNNBuilder* builder);

void CSimpleRNNBuilder_start_new_sequence_with_initial_hidden_states(
    CSimpleRNNBuilder* builder, const CExpression* const h_0[], size_t n);


// ---------------- declarations from lstm.h ----------------

/**
 * SimpleRNNBuilder
 */
CVanillaLSTMBuilder* CVanillaLSTMBuilder_new(unsigned layers,
                                             unsigned input_dim,
                                             unsigned hidden_dim,
                                             CParameterCollection* model,
                                             bool ln_lstm);

void CVanillaLSTMBuilder_delete(CVanillaLSTMBuilder* builder);

void CVanillaLSTMBuilder_start_new_sequence(CVanillaLSTMBuilder* builder);

void CVanillaLSTMBuilder_start_new_sequence_with_initial_hidden_states(
    CVanillaLSTMBuilder* builder, const CExpression* const h_0[], size_t n);

void CVanillaLSTMBuilder_set_dropout(CVanillaLSTMBuilder* builder,
                                     float d,
                                     float d_r);

void CVanillaLSTMBuilder_disable_dropout(CVanillaLSTMBuilder* builder);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // DYNETC_C_API_H_