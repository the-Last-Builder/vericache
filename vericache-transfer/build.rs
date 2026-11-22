use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=CC");
    let out = std::path::PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let cc = env::var_os("CC").unwrap_or_else(|| "cc".into());
    for (name, source) in [("verbs", "src/rdma/verbs.c"), ("uring", "src/nvme/uring.c")] {
        println!("cargo:rerun-if-changed={source}");
        let object = out.join(format!("{name}.o"));
        assert!(
            std::process::Command::new(&cc)
                .args([
                    "-std=c11", "-O2", "-fPIC", "-Wall", "-Werror", "-c", source, "-o"
                ])
                .arg(&object)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            std::process::Command::new("ar")
                .arg("crs")
                .arg(out.join(format!("libtransfer_{name}.a")))
                .arg(object)
                .status()
                .unwrap()
                .success()
        );
        println!("cargo:rustc-link-lib=static=transfer_{name}");
    }
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let cuda_root = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_lib64 = Path::new(&cuda_root).join("lib64");
    if cuda_lib64.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib64.display());
    }
    let cuda_lib = Path::new(&cuda_root).join("lib");
    if cuda_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
}
