extern crate libc;
extern crate dynet_sys as dy;

use std::ops;
// use std::slice;


macro_rules! impl_new {
  ($name:ident, $call:ident) => {
    impl $name {
      pub fn new() -> Self {
        unsafe {
          let inner = dy::$call();
          assert!(!inner.is_null());
          $name {
            inner: inner,
          }
        }
      }
    }
  }
}


macro_rules! impl_drop {
  ($name:ident, $call:ident) => {
    impl Drop for $name {
      fn drop(&mut self) {
        unsafe {
          dy::$call(self.inner);
        }
      }
    }
  }
}


#[derive(Debug)]
pub struct Dim {
    inner: *mut dy::CDim,
}

impl Dim {
    /*
    pub fn new(&) -> Self {
        
    }
    */

    pub fn from_slice(x: &[u64]) -> Self {
        unsafe { Dim { inner: dy::CDim_new_from_array(x.as_ptr() as *const _) } }
    }
}

/*
impl<'a, T> From<&'a [T]> for Dim {
    fn from(x: &'a [T]) -> Self {
        unsafe { Dim { inner: dy::CDim_new_from_array(x.as_ptr() as *const _) } }
    }
}
*/


#[derive(Debug)]
pub struct Tensor {
    inner: *mut dy::CTensor,
}

// impl_drop!(Tensor, CTensor_delete);


pub fn as_scalar(t: Tensor) -> f32 {
    unsafe { dy::C_as_scalar(t.inner) }
}

/*
pub fn as_vector(t: Tensor) -> Vec<f32> {
    // dy::C_as_vector(t.inner).to_vec()
    slice::from_raw_parts_mut(*(dy::C_as_vector(t.inner)
    // Vec::from())
    let slice = unsafe { slice::from_raw_parts(some_pointer, len) };
}
*/

/*
impl<T: TensorType> AsRef<[T]> for Buffer<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data(), (*self.inner).length) }
    }
}

impl<T: TensorType> AsMut<[T]> for Buffer<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.data_mut(), (*self.inner).length) }
    }
}
*/


#[derive(Debug)]
pub struct Parameter {
    inner: *mut dy::CParameter,
}

impl_drop!(Parameter, CParameter_delete);


#[derive(Debug)]
pub struct Expression {
    inner: *mut dy::CExpression,
}


#[derive(Debug)]
pub struct ParameterCollection {
    inner: *mut dy::CParameterCollection,
}

impl_new!(ParameterCollection, CParameterCollection_new);
impl_drop!(ParameterCollection, CParameterCollection_delete);

impl ParameterCollection {
    pub fn add_parameters(&mut self, d: &Dim) -> Parameter {
        unsafe {
            Parameter {
                inner: dy::CParameterCollection_add_parameters(self.inner, d.inner) as *mut _,
            }
        }
    }
}


#[derive(Debug)]
pub struct ComputationGraph {
    inner: *mut dy::CComputationGraph,
}

impl_new!(ComputationGraph, CComputationGraph_new);
impl_drop!(ComputationGraph, CComputationGraph_delete);

impl ComputationGraph {
    pub fn forward(&mut self, last: &Expression) -> Tensor {
        unsafe { Tensor { inner: dy::CComputationGraph_forward(self.inner, last.inner) as *mut _ } }
    }

    pub fn backward(&mut self, last: &Expression) {
        unsafe {
            dy::CComputationGraph_backward(self.inner, last.inner);
        }
    }
}


pub fn input_scalar(g: &mut ComputationGraph, s: f32) -> Expression {
    unsafe { Expression { inner: &mut dy::C_input_scalar(g.inner, s) } }
}

pub fn input_vector(g: &mut ComputationGraph, d: &Dim, data: &Vec<f32>) -> Expression {
    unsafe { Expression { inner: &mut dy::C_input_vector(g.inner, d.inner, data.as_ptr()) } }
}

pub fn parameter(g: &mut ComputationGraph, p: &mut Parameter) -> Expression {
    unsafe { Expression { inner: &mut dy::C_parameter(g.inner, p.inner) } }
}

impl ops::Add for Expression {
    type Output = Expression;

    fn add(self, rhs: Expression) -> Expression {
        unsafe { Expression { inner: &mut dy::C_op_add(self.inner, rhs.inner) } }
    }
}

impl ops::Mul for Expression {
    type Output = Expression;

    fn mul(self, rhs: Expression) -> Expression {
        unsafe { Expression { inner: &mut dy::C_op_mul(self.inner, rhs.inner) } }
    }
}

pub fn tanh(x: &Expression) -> Expression {
    unsafe { Expression { inner: &mut dy::C_tanh(x.inner) } }
}

pub fn squared_distance(x: &Expression, y: &Expression) -> Expression {
    unsafe { Expression { inner: &mut dy::C_squared_distance(x.inner, y.inner) } }
}
