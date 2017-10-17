#ifndef DYNET_C_C_API_H_
#define DYNET_C_C_API_H_


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

typedef float real;

typedef struct CDynetParams CDynetParams;

typedef struct CDim CDim;

typedef struct CTensor CTensor;

typedef struct CParameter CParameter;

typedef struct CLookupParameter CLookupParameter;

typedef struct CParameterCollection CParameterCollection;

typedef unsigned VariableIndex;

typedef struct CComputationGraph CComputationGraph;

typedef struct CExpression CExpression;


// ---------------- declarations from init.h ----------------


// ---------------- declarations from dim.h ----------------

/**
 * Dim
 */
CDim* CDim_new();

CDim* CDim_new_from_array(const long* x);

void CDim_delete(CDim* d);

int CDim_size(CDim* d);


// ---------------- declarations from tensor.h ----------------

/**
 * Tensor
 */


real C_as_scalar(const CTensor* t);

real* C_as_vector(const CTensor* v);


// ---------------- declarations from model.h ----------------

/**
 * Parameter
 */
CParameter* CParameter_new();

void CParameter_delete(CParameter* p);

void CParameter_zero(CParameter* p);

CDim CParameter_dim(CParameter* p);

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

CTensor* CLookupParameter_values(CLookupParameter* p);

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


// ---------------- declarations from expr.h ----------------

CExpression* C_input_scalar(CComputationGraph* g, real s);
CExpression* C_input_vector(CComputationGraph* g, const CDim* d,
                            const float* data);
CExpression* C_parameter(CComputationGraph* g, CParameter* p);
CExpression* C_lookup_parameter(CComputationGraph* g, CLookupParameter* p);

CExpression* C_op_add(const CExpression* x, const CExpression* y);
CExpression* C_op_mul(const CExpression* x, const CExpression* y);

CExpression* C_tanh(const CExpression* x);

CExpression* C_squared_distance(const CExpression* x, const CExpression* y);


#ifdef __cplusplus
} /* end extern "C" */
#endif

void hello();
#endif  // DYNET_C_C_API_H_