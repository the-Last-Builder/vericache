mod ffi;
mod policy;
mod schema;
mod suites;
mod trace;

use clap::Parser;
use schema::JobSpec;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "mooncake-bench",
    about = "Native Rust/C++ Mooncake 0.3.14 TransferEngine benchmark driver (Suites 2, 4, 5)"
)]
struct Cli {
    /// Path to JSON JobSpec file (mutually exclusive with --job-json).
    #[arg(long)]
    job_file: Option<PathBuf>,

    /// Inline JSON JobSpec string.
    #[arg(long)]
    job_json: Option<String>,

    /// Internal subprocess role for 2-process suites ("donor" | "initiator" | "node1").
    #[arg(long)]
    role: Option<String>,

    /// Internal peer TCP address for child subprocess.
    #[arg(long)]
    peer_addr: Option<String>,

    /// Internal session secret for 2-process handshake.
    #[arg(long)]
    session: Option<String>,
}

fn main() {
    let cli = Cli::parse();
    let raw = if let Some(path) = cli.job_file {
        fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read job file {}: {e}", path.display()))
    } else if let Some(json) = cli.job_json {
        json
    } else {
        panic!("must specify --job-file or --job-json");
    };

    let job: JobSpec =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("invalid JobSpec JSON: {e}"));

    if let Some(role) = cli.role.as_deref() {
        let peer_addr = cli.peer_addr.as_deref();
        let session = cli.session.as_deref();
        match job.suite.as_str() {
            "s2_rdma_ablation" => {
                suites::rdma_ablation::run_role(&job, role, peer_addr, session);
            }
            "s4_safety_revoke" => {
                suites::safety_revoke::run_role(&job, role, peer_addr, session);
            }
            "s5_scaleup_cluster" => {
                suites::scaleup_cluster::run_role(&job, role, peer_addr, session);
            }
            other => panic!("unexpected subprocess role '{role}' for suite '{other}'"),
        }
        return;
    }

    let records = match job.suite.as_str() {
        "s2_rdma_ablation" => suites::rdma_ablation::run(&job),
        "s4_safety_revoke" => suites::safety_revoke::run(&job),
        "s5_scaleup_cluster" => suites::scaleup_cluster::run(&job),
        other => panic!("mooncake-bench does not support suite '{other}'"),
    };

    for rec in records {
        let line = serde_json::to_string(&rec).expect("serialize MetricRecord");
        println!("{line}");
    }
}
