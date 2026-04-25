# Plan de Mejoras: Propagación de la Matriz B para Laminados Compuestos Asimétricos

## Contexto y problema

En el análisis aeroelástico de palas de aerogeneradores de eje vertical (VAWT) con materiales laminados, la matriz de acoplamiento membrana–flexión **B** (de la Teoría Clásica de Laminados, CLT) se **ignora silenciosamente** en el pipeline actual.

La estructura `Laminate` de Rust calcula B correctamente, y ambas rutinas de elemento (MITC3, MITC4) la usan cuando es distinta de cero — pero el pipeline la descarta al convertir `PyLaminate → MaterialSpec::Composite → composite_constitutive()`.

**Consecuencia práctica:** los laminados asimétricos (zonas de transición, layups no balanceados, spar caps) producen rigidez incorrecta, frecuencias naturales erróneas y deflexiones estáticas incorrectas bajo carga aerodinámica.

---

## Por qué esta es la prioridad #1 para palas VAWT con compuestos

- Los layups de pala VAWT raramente son simétricos en toda la sección transversal (spar cap, borde de salida y skin tienen apilados distintos).
- El acoplamiento membrana–flexión gobierna el **twist-bend coupling**, que es el mecanismo aeroelástico dominante en palas de eje vertical.
- Sin B, el modelo FEM subestima el alabeo inducido por carga axial centrífuga y sobreestima la rigidez torsional efectiva.

---

## Estado actual verificado

| Componente | Estado |
|---|---|
| Corrección de cortante transversal | ✅ ya implementada (energy-based en `Laminate::compute_shear_stiffness`) |
| Uso de B en las rutinas de elemento (MITC3, MITC4) | ✅ implementado — ambos comprueban `coupling.amax()` e integran K_coupling |
| Updated Lagrangian (`update_reference`) | ✅ integrado en el pipeline FSI |
| Log strain vs. Green-Lagrange | No es brecha — cepas de membrana < 1% en VAWT, diferencia despreciable |
| Offset de superficie de referencia | No es brecha para VAWT — la superficie de referencia coincide con la midsurface |

---

## Causa raíz: un único punto de ruptura en el pipeline

```
Laminate.b  (calculado correctamente en compute_abd_matrices)
    ↓
compute_composite_spec()           [aeroelast-py/src/lib.rs ~línea 1540]
    → extrae corrected_lam.a, .d, .cs
    → descarta corrected_lam.b               ← AQUÍ SE PIERDE
    → construye MaterialSpec::Composite { cm, cb, cs, … }  (sin campo B)
    ↓
build_constitutive_mitc3/4()       [assembly/assembler.rs ~línea 1372]
    → llama composite_constitutive(cm, cb, cs, h)
    ↓
composite_constitutive()           [materials/composite.rs línea 48]
    → coupling: Matrix3::zeros()             ← B = 0 siempre
```

---

## Cambios requeridos

### 1. `crates/aeroelast-core/src/materials/composite.rs`

Agregar parámetro `b_flat: &[f64; 9]` y usarlo en lugar de ceros:

```rust
pub fn composite_constitutive(
    a_flat:  &[f64; 9],
    b_flat:  &[f64; 9],   // ← NUEVO
    d_flat:  &[f64; 9],
    cs_flat: &[f64; 4],
    thickness: f64,
) -> ShellConstitutive {
    // construir cm, cb, cs igual que antes ...
    let coupling = Matrix3::new(
        b_flat[0], b_flat[1], b_flat[2],
        b_flat[3], b_flat[4], b_flat[5],
        b_flat[6], b_flat[7], b_flat[8],
    );
    ShellConstitutive { cm, cb, cs, cm_raw, coupling }
}
```

---

### 2. `crates/aeroelast-core/src/assembly/assembler.rs`

Agregar campo `b: [f64; 9]` a la variante `MaterialSpec::Composite` y pasarlo en `build_constitutive_mitc3/4()`:

```rust
Composite {
    cm: [f64; 9],
    b:  [f64; 9],   // ← NUEVO — B coupling matrix (CLT)
    cb: [f64; 9],
    cs: [f64; 4],
    thickness: f64,
    e_equiv: f64,
    mass_per_area: f64,
    rotational_inertia: f64,
},
```

En `build_constitutive_mitc3/4()`:

```rust
MaterialSpec::Composite { cm, b, cb, cs, thickness, .. } =>
    composite_constitutive(cm, b, cb, cs, *thickness),
```

---

### 3. `crates/aeroelast-py/src/lib.rs`

**a) Conversión `PyLaminate → MaterialSpec::Composite`** (función `compute_composite_spec`, ~línea 1540):

Extraer `corrected_lam.b` y poblarlo en el struct:

```rust
let mut b_arr = [0.0f64; 9];
for ii in 0..3 {
    for jj in 0..3 {
        b_arr[ii * 3 + jj] = corrected_lam.b[(ii, jj)];
    }
}
Ok(MaterialSpec::Composite { cm, b: b_arr, cb, cs, … })
```

**b) Path de dict (`parse_material()`, ~línea 1190)**:

Agregar clave opcional `"b"` (lista de 9 floats). Por defecto `[0.0; 9]` para compatibilidad hacia atrás:

```rust
let b_arr = extract_optional_mat3(dict, "b").unwrap_or([0.0; 9]);
MaterialSpec::Composite { cm, b: b_arr, cb, cs, … }
```

---

### 4. `src/aeroelast/core/assembler.py`

No se requiere cambio para el path `PyLaminate` (el bridge Rust lo maneja). Para el path de dict, documentar que `"b"` es una clave opcional aceptada (lista de 9 floats, row-major). Los callers existentes sin `"b"` no se rompen.

---

## Archivos a modificar

| Archivo | Cambio |
|---|---|
| `crates/aeroelast-core/src/materials/composite.rs` | Agregar param `b_flat`, usar en lugar de zeros |
| `crates/aeroelast-core/src/assembly/assembler.rs` | Agregar `b: [f64; 9]` a `MaterialSpec::Composite`; pasar en build_constitutive |
| `crates/aeroelast-py/src/lib.rs` | Extraer B de `corrected_lam.b` en `compute_composite_spec`; clave `"b"` en `parse_material` |

---

## Fuera de scope (en esta iteración)

| Mejora | Por qué se pospone |
|---|---|
| **Offset de superficie de referencia** | `to_shell_constitutive_with_offset()` ya existe; threadear `z_offset` por todo el stack es un refactor mayor. No necesario para VAWT donde ref surface = midsurface. |
| **Log strain (Hencky)** | Cepas de membrana en palas VAWT < 1%; GL y log strain son idénticos en ese régimen. |
| **B·κ en K_sigma (rigidez geométrica)** | Corrección de segundo orden; solo importa a grandes deformaciones de flexión. Abordable después de fijar la propagación básica. |

---

## Verificación

### 1. Regresión — tests existentes deben pasar sin cambios

```bash
pytest tests/test_large_rotation_benchmarks.py tests/test_mitc3_benchmarks.py
```

Los laminados isotrópicos tienen B = 0 → comportamiento idéntico al actual. Los 15 tests deben seguir pasando.

### 2. Smoke test B-coupling

Laminado asimétrico `[0/90]` (un ply a 0° arriba, uno a 90° abajo, sin simetría): bajo carga axial de membrana debe aparecer flexión transversal (w ≠ 0). Con el código actual, w = 0.

### 3. Test cuantitativo

Strip [0/90] con propiedades típicas de CFRP (E1 = 181 GPa, E2 = 10.3 GPa, G12 = 7.17 GPa, ν12 = 0.28, h = 1 mm por ply): verificar que la deflexión en punta calculada por el elemento coincide con la predicción analítica CLT dentro del 2%.
