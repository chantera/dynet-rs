/* automatically generated by rust-bindgen */

pub const __bool_true_false_are_defined: ::std::os::raw::c_uint = 1;
pub const true_: ::std::os::raw::c_uint = 1;
pub const false_: ::std::os::raw::c_uint = 0;
pub type bool_ = ::std::os::raw::c_uchar;
pub type real = f32;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CDynetParams {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CDim {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CTensor {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CParameter {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CLookupParameter {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CParameterCollection {
    _unused: [u8; 0],
}
pub type VariableIndex = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CComputationGraph {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CExpression {
    _unused: [u8; 0],
}
extern "C" {
    /// Dim
    pub fn CDim_new() -> *mut CDim;
}
extern "C" {
    pub fn CDim_new_from_array(x: *const ::std::os::raw::c_long) -> *mut CDim;
}
extern "C" {
    pub fn CDim_delete(d: *mut CDim);
}
extern "C" {
    pub fn CDim_size(d: *mut CDim) -> ::std::os::raw::c_int;
}
extern "C" {
    /// Tensor
    pub fn C_as_scalar(t: *const CTensor) -> real;
}
extern "C" {
    pub fn C_as_vector(v: *const CTensor) -> *mut real;
}
extern "C" {
    /// Parameter
    pub fn CParameter_new() -> *mut CParameter;
}
extern "C" {
    pub fn CParameter_delete(p: *mut CParameter);
}
extern "C" {
    pub fn CParameter_zero(p: *mut CParameter);
}
extern "C" {
    pub fn CParameter_dim(p: *mut CParameter) -> CDim;
}
extern "C" {
    pub fn CParameter_values(p: *mut CParameter) -> *mut CTensor;
}
extern "C" {
    pub fn CParameter_set_updated(p: *mut CParameter, b: bool_);
}
extern "C" {
    pub fn CParameter_is_updated(p: *mut CParameter) -> bool_;
}
extern "C" {
    /// LookupParameter
    pub fn CLookupParameter_new() -> *mut CLookupParameter;
}
extern "C" {
    pub fn CLookupParameter_delete(p: *mut CLookupParameter);
}
extern "C" {
    pub fn CLookupParameter_zero(p: *mut CLookupParameter);
}
extern "C" {
    pub fn CLookupParameter_dim(p: *mut CLookupParameter) -> *mut CDim;
}
extern "C" {
    pub fn CLookupParameter_values(p: *mut CLookupParameter) -> *mut CTensor;
}
extern "C" {
    pub fn CLookupParameter_set_updated(p: *mut CLookupParameter, b: bool_);
}
extern "C" {
    pub fn CLookupParameter_is_updated(p: *mut CLookupParameter) -> bool_;
}
extern "C" {
    /// ParameterCollection
    pub fn CParameterCollection_new() -> *mut CParameterCollection;
}
extern "C" {
    pub fn CParameterCollection_delete(pc: *mut CParameterCollection);
}
extern "C" {
    pub fn CParameterCollection_gradient_l2_norm(pc:
                                                     *mut CParameterCollection)
     -> f32;
}
extern "C" {
    pub fn CParameterCollection_reset_gradient(pc: *mut CParameterCollection);
}
extern "C" {
    pub fn CParameterCollection_add_parameters(pc: *mut CParameterCollection,
                                               d: *const CDim)
     -> *mut CParameter;
}
extern "C" {
    pub fn CParameterCollection_add_lookup_parameters(pc:
                                                          *mut CParameterCollection,
                                                      n:
                                                          ::std::os::raw::c_uint,
                                                      d: *const CDim)
     -> *mut CLookupParameter;
}
extern "C" {
    /// ComputationGraph
    pub fn CComputationGraph_new() -> *mut CComputationGraph;
}
extern "C" {
    pub fn CComputationGraph_delete(g: *mut CComputationGraph);
}
extern "C" {
    pub fn CComputationGraph_forward(g: *mut CComputationGraph,
                                     last: *const CExpression)
     -> *const CTensor;
}
extern "C" {
    pub fn CComputationGraph_backward(g: *mut CComputationGraph,
                                      last: *const CExpression);
}
extern "C" {
    pub fn C_input_scalar(g: *mut CComputationGraph, s: real)
     -> *mut CExpression;
}
extern "C" {
    pub fn C_input_vector(g: *mut CComputationGraph, d: *const CDim,
                          data: *const f32) -> *mut CExpression;
}
extern "C" {
    pub fn C_parameter(g: *mut CComputationGraph, p: *mut CParameter)
     -> *mut CExpression;
}
extern "C" {
    pub fn C_lookup_parameter(g: *mut CComputationGraph,
                              p: *mut CLookupParameter) -> *mut CExpression;
}
extern "C" {
    pub fn C_op_add(x: *const CExpression, y: *const CExpression)
     -> *mut CExpression;
}
extern "C" {
    pub fn C_op_mul(x: *const CExpression, y: *const CExpression)
     -> *mut CExpression;
}
extern "C" {
    pub fn C_tanh(x: *const CExpression) -> *mut CExpression;
}
extern "C" {
    pub fn C_squared_distance(x: *const CExpression, y: *const CExpression)
     -> *mut CExpression;
}
extern "C" {
    pub fn hello();
}
