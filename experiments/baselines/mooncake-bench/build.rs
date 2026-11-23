use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_mooncake_dir() -> PathBuf {
    if let Ok(dir) = env::var("MOONCAKE_LIB_DIR") {
        let p = PathBuf::from(dir);
        if p.join("libmooncake_pg.so").exists() {
            return p;
        }
    }
    let candidates = [
        "/opt/vericache/bench-venv/lib/python3.12/site-packages/mooncake",
        "/usr/local/lib/python3.12/dist-packages/mooncake",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.join("libmooncake_pg.so").exists() {
            return p;
        }
    }
    panic!(
        "Could not locate libmooncake_pg.so. Set MOONCAKE_LIB_DIR or build inside vericache-bench container."
    );
}

fn main() {
    println!("cargo:rerun-if-changed=cpp/mooncake_ffi.cpp");
    println!("cargo:rerun-if-env-changed=MOONCAKE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let cuda_inc = Path::new(&cuda_home).join("include");
    let cuda_lib = Path::new(&cuda_home).join("lib64");
    let mc_dir = find_mooncake_dir();
    let mc_libs_dir = mc_dir
        .parent()
        .expect("mooncake parent")
        .join("mooncake_transfer_engine_cuda13.libs");

    let obj_path = out_dir.join("mooncake_ffi.o");
    let lib_path = out_dir.join("libmooncake_ffi.a");

    let status = Command::new("g++")
        .args([
            "-O3",
            "-std=c++20",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-c",
            "cpp/mooncake_ffi.cpp",
            "-I",
            cuda_inc.to_str().unwrap(),
            "-o",
            obj_path.to_str().unwrap(),
        ])
        .status()
        .expect("failed to invoke g++");
    assert!(status.success(), "g++ failed compiling cpp/mooncake_ffi.cpp");

    let ar_status = Command::new("ar")
        .args(["crs", lib_path.to_str().unwrap(), obj_path.to_str().unwrap()])
        .status()
        .expect("failed to invoke ar");
    assert!(ar_status.success(), "ar failed building libmooncake_ffi.a");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-search=native={}", mc_dir.display());
    println!("cargo:rustc-link-lib=static=mooncake_ffi");
    println!("cargo:rustc-link-lib=dylib=mooncake_pg");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    let rpath = format!(
        "{}:{}:{}",
        mc_dir.display(),
        mc_libs_dir.display(),
        cuda_lib.display()
    );
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rpath}");
}
