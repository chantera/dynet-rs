extern crate libc;
extern crate dynet_sys as dy;


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
}

impl<'a> From<&'a [i64]> for Dim {
    fn from(x: &'a [i64]) -> Self {
        unsafe { Dim { inner: dy::CDim_new_from_array(x.as_ptr() as *const _) } }
    }
}


#[derive(Debug)]
pub struct Tensor {
    inner: *mut dy::CTensor,
}

// impl_drop!(Tensor, CTensor_delete);


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
}


pub fn input(g: &mut ComputationGraph, s: f32) -> Expression {
    unsafe { Expression { inner: &mut dy::C_input(g.inner, s) } }
}

pub fn parameter(g: &mut ComputationGraph, p: &mut Parameter) -> Expression {
    unsafe { Expression { inner: &mut dy::C_parameter(g.inner, p.inner) } }
}

pub fn squared_distance(x: &Expression, y: &Expression) -> Expression {
    unsafe { Expression { inner: &mut dy::C_squared_distance(x.inner, y.inner) } }
}
