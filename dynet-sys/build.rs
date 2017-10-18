extern crate bindgen;

use std::env;
use std::error::Error;
use std::result::Result;
use std::process::exit;

fn main() {
    exit(match build() {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}

fn build() -> Result<(), Box<Error>> {
    let lib_dir = try!(env::var("DYNET_LIBRARY_DIR").map_err(|e| {
        format!(
            "{}: Run with `DYNET_LIBRARY_DIR=/path/to/lib`",
            e.to_string()
        )
    }));
    let include_dir = try!(env::var("DYNET_INCLUDE_DIR").map_err(|e| {
        format!(
            "{}: Run with `DYNET_INCLUDE_DIR=/path/to/include`",
            e.to_string()
        )
    }));

    println!("cargo:rustc-link-lib=static=dynetc");
    println!("cargo:rustc-link-search={}", lib_dir);
    // println!("cargo:include={}", include_dir);

    let bindings = bindgen::Builder::default()
        .header("../c/dynetc/c_api.h")
        // .clang_arg("-I../c")
        .rustfmt_bindings(false)
        .generate_comments(false)
        .generate()
        .expect("Unable to generate bindings");

    bindings.write_to_file("src/bindings.rs").expect(
        "Couldn't write bindings!",
    );
    Ok(())
}
