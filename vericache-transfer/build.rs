use std::env;
use std::path::Path;

fn main() {
    let rdma = env::var_os("CARGO_FEATURE_RDMA").is_some();
    let nvme = env::var_os("CARGO_FEATURE_NVME").is_some();
    if rdma || nvme {
        println!("cargo:rerun-if-env-changed=CC");
        let out = std::path::PathBuf::from(env::var_os("OUT_DIR").unwrap());
        let cc = env::var_os("CC").unwrap_or_else(|| "cc".into());
        for (enabled, name, source) in [
            (rdma, "verbs", "src/rdma/verbs.c"),
            (nvme, "uring", "src/nvme/uring.c"),
        ] {
            if !enabled {
                continue;
            }
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
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");

        let cuda_root = env::var("CUDA_HOME")
            .or_else(|_| env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        for dir in ["lib64", "lib"] {
            let lib = Path::new(&cuda_root).join(dir);
            if lib.exists() {
                println!("cargo:rustc-link-search=native={}", lib.display());
            }
        }
    }
}
