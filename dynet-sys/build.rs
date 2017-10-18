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
    let lib_dir = env::var("DYNET_LIBRARY_DIR").unwrap_or("../c/build/dynetc".to_string());
    println!("cargo:rustc-link-lib=dylib=dynetc");
    println!("cargo:rustc-link-search={}", lib_dir);

    let bindings = bindgen::Builder::default()
        .header("../c/dynetc/c_api.h")
        .rustfmt_bindings(false)
        .generate_comments(false)
        .generate()
        .expect("Unable to generate bindings");

    bindings.write_to_file("src/bindings.rs").expect(
        "Couldn't write bindings!",
    );
    Ok(())
}
