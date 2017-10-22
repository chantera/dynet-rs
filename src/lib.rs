extern crate libc;
extern crate dynet_sys as dy;

use libc::c_char;
use libc::c_int;
use std::ops;
use std::ffi::CString;
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


pub fn initialize() {
    let args = std::env::args()
        .map(|arg| CString::new(arg).unwrap())
        .collect::<Vec<CString>>();
    let c_args = args.iter()
        .map(|arg| arg.as_ptr() as *mut _)
        .collect::<Vec<*mut c_char>>();
    unsafe {
        dy::C_initialize(c_args.len() as c_int, c_args.as_ptr() as *mut _);
    }
}


#[derive(Debug)]
pub struct Dim {
    inner: *mut dy::CDim,
}

impl Dim {
    pub fn from_slice(x: &[u64]) -> Self {
        unsafe { Dim { inner: dy::CDim_new_from_array(x.as_ptr() as *const _, x.len()) } }
    }

    pub fn size(&self) -> u32 {
        unsafe { dy::CDim_size(self.inner) }
    }

    pub fn rows(&self) -> u32 {
        unsafe { dy::CDim_rows(self.inner) }
    }

    pub fn cols(&self) -> u32 {
        unsafe { dy::CDim_cols(self.inner) }
    }

    pub fn batch_elems(&self) -> u32 {
        unsafe { dy::CDim_batch_elems(self.inner) }
    }

    pub fn shape(&self) -> Vec<u32> {
        unsafe {
            let nd = dy::CDim_ndims(self.inner);
            let mut shape = Vec::with_capacity(nd as usize);
            for i in 0..nd {
                shape.push(dy::CDim_dim_size(self.inner, i));
            }
            // if force_batch or d.batch_elems() > 1 : dim.append(d.batch_elems())
            shape
        }
    }
}

impl_new!(Dim, CDim_new);
impl_drop!(Dim, CDim_delete);

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
    // dim: Dim,
}

impl Tensor {
    // fn from_raw(t: *mut dy::CTensor) -> Self {
    //     unsafe {
    //         Tensor {
    //             inner: t,
    //             // dim: Dim { inner: dy::CTensor_dim(t) },
    //         }
    //     }
    // }

    pub fn dim(&self) -> Dim {
        unsafe { Dim { inner: dy::CTensor_dim(self.inner) } }
        // &self.dim
    }
}

impl_drop!(Tensor, CTensor_delete);


pub fn as_scalar(t: &Tensor) -> f32 {
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
    pub expr: dy::CExpression,
    // pub inner: *mut dy::CExpression,
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
        unsafe {
            Tensor { inner: dy::CComputationGraph_forward(self.inner, &last.expr) as *mut _ }
            // Tensor::from_raw(dy::CComputationGraph_forward(self.inner, last.inner) as
            //     *mut _)
        }
    }

    pub fn backward(&mut self, last: &Expression) {
        unsafe {
            dy::CComputationGraph_backward(self.inner, &last.expr);
        }
    }
}


pub trait Trainer {
    fn update(&mut self);
}


#[derive(Debug)]
pub struct SimpleSGDTrainer {
    inner: *mut dy::CSimpleSGDTrainer,
}

impl_drop!(SimpleSGDTrainer, CSimpleSGDTrainer_delete);

impl SimpleSGDTrainer {
    pub fn new(m: &mut ParameterCollection, learning_rate: f32) -> Self {
        unsafe {
            let inner = dy::CSimpleSGDTrainer_new(m.inner, learning_rate);
            assert!(!inner.is_null());
            SimpleSGDTrainer { inner: inner }
        }
    }
}

impl Trainer for SimpleSGDTrainer {
    fn update(&mut self) {
        unsafe { dy::CTrainer_update(self.inner as *mut _) }
    }
}


pub fn input_scalar(g: &mut ComputationGraph, s: f32) -> Expression {
    unsafe { Expression { expr: dy::C_input_scalar(g.inner, s) } }
}

pub fn input_vector(g: &mut ComputationGraph, d: &Dim, data: &Vec<f32>) -> Expression {
    unsafe { Expression { expr: dy::C_input_vector(g.inner, d.inner, data.as_ptr(), data.len()) } }
}

pub fn parameter(g: &mut ComputationGraph, p: &mut Parameter) -> Expression {
    unsafe { Expression { expr: dy::C_parameter(g.inner, p.inner) } }
}

impl ops::Add for Expression {
    type Output = Self;

    fn add(self, rhs: Expression) -> Expression {
        unsafe { Expression { expr: dy::C_op_add(&self.expr, &rhs.expr) } }
    }
}

impl ops::Mul for Expression {
    type Output = Self;

    fn mul(self, rhs: Expression) -> Expression {
        unsafe { Expression { expr: dy::C_op_mul(&self.expr, &rhs.expr) } }
    }
}

pub fn tanh(x: &Expression) -> Expression {
    unsafe { Expression { expr: dy::C_tanh(&x.expr) } }
}

pub fn squared_distance(x: &Expression, y: &Expression) -> Expression {
    unsafe { Expression { expr: dy::C_squared_distance(&x.expr, &y.expr) } }
}
