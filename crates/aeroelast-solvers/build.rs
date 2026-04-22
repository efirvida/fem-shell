fn main() {
    // ── MPI (OpenMPI 4.1.4) ──────────────────────────────────────────────────
    // libpetsc.so depends on libmpi.so.40 from the system OpenMPI install.
    // pkg-config for PETSc does not emit -lmpi, so we add it explicitly here.
    println!("cargo:rustc-link-search=native=/scratch/app/openmpi/4.1.4_gnu/lib");
    println!("cargo:rustc-link-lib=mpi");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/scratch/app/openmpi/4.1.4_gnu/lib");

    // ── PETSc ────────────────────────────────────────────────────────────────
    let petsc_libs = std::process::Command::new("pkg-config")
        .args(["--libs-only-L", "PETSc"])
        .output()
        .expect("pkg-config --libs-only-L PETSc failed. Is PETSc installed?");

    let petsc_link = std::process::Command::new("pkg-config")
        .args(["--libs-only-l", "PETSc"])
        .output()
        .expect("pkg-config --libs-only-l PETSc failed.");

    let petsc_rpath = std::process::Command::new("pkg-config")
        .args(["--variable=libdir", "PETSc"])
        .output()
        .expect("pkg-config --variable=libdir PETSc failed.");

    emit_link_flags(&petsc_libs.stdout);
    emit_lib_flags(&petsc_link.stdout);

    // rpath so the binary finds libpetsc.so at runtime
    let rpath = String::from_utf8_lossy(&petsc_rpath.stdout)
        .trim()
        .to_string();
    if !rpath.is_empty() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{rpath}");
    }

    // ── SLEPc ────────────────────────────────────────────────────────────────
    let slepc_libs = std::process::Command::new("pkg-config")
        .args(["--libs-only-L", "SLEPc"])
        .output()
        .expect("pkg-config --libs-only-L SLEPc failed. Is SLEPc installed?");

    let slepc_link = std::process::Command::new("pkg-config")
        .args(["--libs-only-l", "SLEPc"])
        .output()
        .expect("pkg-config --libs-only-l SLEPc failed.");

    let slepc_rpath = std::process::Command::new("pkg-config")
        .args(["--variable=libdir", "SLEPc"])
        .output()
        .expect("pkg-config --variable=libdir SLEPc failed.");

    emit_link_flags(&slepc_libs.stdout);
    emit_lib_flags(&slepc_link.stdout);

    let rpath = String::from_utf8_lossy(&slepc_rpath.stdout)
        .trim()
        .to_string();
    if !rpath.is_empty() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{rpath}");
    }

    // ── preCICE (feature = fsi) ───────────────────────────────────────────────
    // Only link if the `fsi` feature is active.
    if std::env::var("CARGO_FEATURE_FSI").is_ok() {
        let precice_lib_dir = std::env::var("PRECICE_LIB_DIR")
            .unwrap_or_else(|_| "/scratch/leahk/eduardo.donestevez/fem-shell/.sources/build/precice/build".to_string());
        println!("cargo:rustc-link-search=native={precice_lib_dir}");
        println!("cargo:rustc-link-lib=dylib=precice");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{precice_lib_dir}");
        println!("cargo:rerun-if-env-changed=PRECICE_LIB_DIR");
    }

    // Re-run build.rs if pkg-config output changes
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
}

/// Parse `-L/path/to/lib` flags and emit `cargo:rustc-link-search=native=...`
fn emit_link_flags(output: &[u8]) {
    let s = String::from_utf8_lossy(output);
    for token in s.split_whitespace() {
        if let Some(path) = token.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        }
    }
}

/// Parse `-lfoo` flags and emit `cargo:rustc-link-lib=foo`
fn emit_lib_flags(output: &[u8]) {
    let s = String::from_utf8_lossy(output);
    for token in s.split_whitespace() {
        if let Some(lib) = token.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}
