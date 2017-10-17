extern crate bindgen;

use std::env;

fn main() {
    // println!("cargo:rustc-link-lib=dylib=dynet");
    // println!("cargo:rustc-link-search={}", env::var("DYNET_INCLUDE_DIR"));

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        // .clang_arg("-I/../c")
        .clang_arg("-I/Users/hiroki/Desktop/dynet-rs/c")
        .clang_arg(format!("-I/{}", env::var("DYNET_INCLUDE_DIR").unwrap()))
        // // .clang_arg(format!("-I/{}", env::var("EIGEN_INCLUDE_DIR").unwrap()))
        // .clang_arg("-mmacosx-version-min=10.7")
        // .clang_arg("-x")
        // .clang_arg("c++")
        // .clang_arg("-std=c++11")
        // .clang_arg("-stdlib=libc++")
        // .use_core()
        // .raw_line(r#"extern crate core;"#)
        // // .opaque_type("std::.*")
        // .opaque_type("std::string")
        // .rustfmt_bindings(true)
        .generate()
        .expect("Unable to generate bindings");

    bindings.write_to_file("src/bindings.rs").expect(
        "Couldn't write bindings!",
    );
}
