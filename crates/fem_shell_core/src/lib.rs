use pyo3::prelude::*;

// Re-export everything from aeroelast_py so Python imports as `fem_shell_core`
#[pymodule]
fn fem_shell_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    aeroelast_py::register_module(m)?;
    Ok(())
}
