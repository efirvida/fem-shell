# S4R: Elemento Shell de 4 Nodos con Materiales Compuestos Laminados

**Fuente primaria**: Abaqus Theory Guide (STM) 2016, secciones 3.6.1, 3.6.5 y 3.6.8  
URL: `https://ceae-server.colorado.edu/v2016/books/stm/`

> **Nota de limitación**: Las ecuaciones en la documentación original están embebidas como imágenes GIF.
> Este documento reconstruye la formulación a partir del texto descriptivo del manual y la
> teoría estándar de referencia. Donde la formulación proviene de la Teoría Clásica de Laminados
> (CLT) estándar —no del STM de Abaqus directamente— se indica explícitamente con `[CLT]`.

---

## Elementos cubiertos

- **S4R** — shell cuadrilateral de 4 nodos, integración reducida (1 punto Gauss en plano)
- **S4** — ídem con integración completa (2×2 puntos en plano, sin modos espurios)
- **S3R** — triangular de 3 nodos (collapso de S4R)

Todos implementan teoría de Mindlin-Reissner de primer orden (shear-flexible, finite-strain).
Los elementos de 5 grados de libertad (S4R5, S8R5) tienen cinemática diferente y **no** se cubren aquí.

---

## 1. Convenciones de Notación

| Símbolo | Significado |
|---|---|
| `ξ¹, ξ²` | Coordenadas paramétricas de superficie `∈ [-1, 1]` |
| `ζ` | Coordenada de espesor normalizada `∈ [-0.5, 0.5]` |
| `α, β ∈ {1,2}` | Índices de superficie (convenio de suma implícita) |
| `i, j, k ∈ {1,2,3}` | Índices cartesianos globales |
| `I ∈ {1..4}` | Índice de nodo del elemento |
| `N^I(ξ¹,ξ²)` | Función de forma bilineal para nodo I |
| `h` | Espesor total de la sección |
| `n̂` | Normal unitaria a la superficie de referencia |
| `G, g` | Métrica en configuración de referencia y deformada respectivamente |

---

## 2. Geometría e Interpolación (Sección 3.6.1)

### 2.1 Funciones de forma bilineales (S4R)

```
N¹ = (1 - ξ¹)(1 - ξ²) / 4
N² = (1 + ξ¹)(1 - ξ²) / 4
N³ = (1 + ξ¹)(1 + ξ²) / 4
N⁴ = (1 - ξ¹)(1 + ξ²) / 4
```

### 2.2 Posición en el espacio 3D

La posición de cualquier punto del elemento se parametriza como:

```
x(ξ¹, ξ², ζ) = x_ref(ξ¹, ξ²) + ζ · h · n̂(ξ¹, ξ²)

// donde:
x_ref(ξ¹, ξ²) = Σ_I  N^I(ξ¹, ξ²) · X^I     // superficie de referencia
n̂(ξ¹, ξ²)    = Σ_I  N^I(ξ¹, ξ²) · n̂^I     // normal interpolada (aprox.)
```

> La derivada del factor de espesor respecto a ζ se desprecia, simplificando la cinemática.

### 2.3 Sistema de coordenadas local ortonormal

En cada punto de integración se construye un frame local `{ê₁, ê₂, ê₃}`:

```
// ê₃ = normal a superficie de referencia
t₁ = ∂x_ref / ∂ξ¹        // tangente en dirección ξ¹
t₂ = ∂x_ref / ∂ξ²        // tangente en dirección ξ²
ê₃ = normalize(t₁ × t₂)  // normal
ê₁ = normalize(t₁)
ê₂ = ê₃ × ê₁             // completa la base derecha
```

Este frame rota con el elemento (frame corrotacional). Las tensiones y deformaciones
se expresan en este sistema local.

### 2.4 Grados de libertad por nodo

Cada nodo tiene **6 DOF**: 3 translaciones `(u₁, u₂, u₃)` y 3 rotaciones `(ω₁, ω₂, ω₃)`.

> En el plano de la superficie solo son independientes 2 rotaciones (las que cambian n̂).
> La rotación alrededor de n̂ (drill rotation) se controla con una penalización.

---

## 3. Cinemática: Medidas de Deformación (Sección 3.6.5)

### 3.1 Gradiente de deformación incremental

Para análisis finito-strain, el incremento de deformación se calcula desde el gradiente
de deformación incremental:

```
F_inc = F_end · F_begin⁻¹    // gradiente entre inicio y fin del incremento

// Descomposición polar:
F_inc = R_inc · U_inc
// R_inc: rotación incremental
// U_inc: estiramiento incremental (simétrico)
```

### 3.2 Deformación de membrana (logarítmica)

La deformación de membrana logarítmica sigue de la descomposición polar:

```
ε_αβ^membrane = (1/2) · d(ln U²)_αβ

// Para incrementos:
Δε_αβ^membrane = (1/2) · [ΔF_αβ + ΔF_βα] - (1/2)[ΔF_αγ ΔF_γβ + ΔF_βγ ΔF_γα]
// términos de orden 2 necesarios para consistencia a large strain
```

La métrica de la superficie deformada es:
```
g_αβ = (∂x/∂ξ^α) · (∂x/∂ξ^β)
// ε_αβ = (1/2)(g_αβ - G_αβ)   // medida de Green-Lagrange equivalente
```

### 3.3 Cambio de curvatura (Teoría Koiter-Sanders)

```
κ_αβ = κ̄_αβ(configuración deformada) - κ_αβ(configuración referencia)

// Curvatura aproximada (segunda forma fundamental):
κ_αβ ≈ (∂²x / ∂ξ^α∂ξ^β) · n̂

// Incremento de curvatura:
Δκ_αβ = ∂(Δn̂)/∂ξ^α · ∂x/∂ξ^β + ∂x/∂ξ^α · ∂(Δn̂)/∂ξ^β + ...
// La curvatura en el inicio del incremento NO aparece (simplificación válida para
// curvatura inicial moderada)
```

### 3.4 Actualización de la normal con cuaterniones

Para rotaciones finitas, la normal n̂ se actualiza exactamente vía álgebra de cuaterniones:

```
// Cuaternión de rotación incremental q = [q₀, q₁, q₂, q₃]:
q₀ = cos(|θ|/2)
qᵢ = sin(|θ|/2) · θᵢ / |θ|    // θ = vector de rotación incremental

// Actualización:
n̂_new = R(q) · n̂_old

// R(q) como matriz de rotación 3×3:
R = I + (2q₀) · [q̂] + 2 · [q̂]²
// [q̂] = skew-symmetric matrix de la parte vectorial de q
```

---

## 4. Deformación de Cortante Transversal: Método ANS

El elemento S4R utiliza **Assumed Natural Strains (ANS)** basado en el principio de
Hu-Washizu para evitar *shear locking* en problemas de flexión dominante.

### 4.1 Evaluación en puntos de arista (midpoints)

Las deformaciones de cortante transversal no se evalúan en el punto de Gauss central
sino en los midpoints de las cuatro aristas:

```
// Puntos de evaluación (coordenadas paramétricas):
A = ( 0, -1)  // midpoint arista ξ² = -1
B = ( 1,  0)  // midpoint arista ξ¹ = +1
C = ( 0, +1)  // midpoint arista ξ² = +1
D = (-1,  0)  // midpoint arista ξ¹ = -1

// Deformación covariante en cada punto:
γ̄₁^(A) = (∂x/∂ξ¹)|_A · n̂|_A     // cortante ξ¹-dirección en A
γ̄₂^(B) = (∂x/∂ξ²)|_B · n̂|_B     // cortante ξ²-dirección en B
// ídem para C y D
```

### 4.2 Campo asumido interpolado

```
// Componente homogénea (evaluada en el centro del elemento):
γ̄_α^hom = γ̄_α|_(ξ¹=0, ξ²=0)

// Modos adicionales de cortante para estabilización:
γ̄_α^butterfly  // modo mariposa: bending cruzado entre nodos opuestos
γ̄_α^crop_circle // modo circular: barrido de normales

// Campo total:
γ̄_α(ξ¹, ξ²) = γ̄_α^hom + c₁(geom) · γ̄_α^butterfly + c₂(geom) · γ̄_α^crop_circle

// Para geometría con Jacobiano constante: c₁ = c₂ = 1
// Para elementos distorsionados: c₁, c₂ dependen de la geometría
```

> `γ̄_α^butterfly` → variación cruzada de curvatura (bending no uniforme)  
> `γ̄_α^crop_circle` → rotación de la normal en patrón circular (evita locking torsional)

---

## 5. Sección de Shell con Composite Laminado

### 5.1 Definición del laminado

Un laminado de N capas se define por:

```
// Para cada capa k = 1..N:
Layer k {
    z_bottom_k   // coordenada Z inferior de la capa (desde midsurface)
    z_top_k      // coordenada Z superior de la capa
    theta_k      // ángulo de orientación de fibra en el plano (grados)
    E1_k, E2_k  // módulos en dirección de fibra y transversal
    nu12_k       // coeficiente de Poisson
    G12_k        // módulo de corte en plano
    G13_k, G23_k // módulos de corte transversal (para cortante fuera del plano)
}
```

### 5.2 Rigidez reducida de una lámina (Plano-Tensión) `[CLT]`

Para una lámina en sus ejes principales de material:

```
Q11 = E1 / (1 - nu12 · nu21)
Q22 = E2 / (1 - nu12 · nu21)
Q12 = nu12 · E2 / (1 - nu12 · nu21)
Q66 = G12

// donde: nu21 = nu12 · E2 / E1
```

Rotación al sistema de referencia del elemento a ángulo theta:

```
m = cos(theta),  n = sin(theta)

Q̄₁₁ = Q11·m⁴ + 2(Q12 + 2Q66)·m²n² + Q22·n⁴
Q̄₂₂ = Q11·n⁴ + 2(Q12 + 2Q66)·m²n² + Q22·m⁴
Q̄₁₂ = (Q11 + Q22 - 4Q66)·m²n² + Q12(m⁴ + n⁴)
Q̄₁₆ = (Q11 - Q12 - 2Q66)·m³n - (Q22 - Q12 - 2Q66)·mn³
Q̄₂₆ = (Q11 - Q12 - 2Q66)·mn³ - (Q22 - Q12 - 2Q66)·m³n
Q̄₆₆ = (Q11 + Q22 - 2Q12 - 2Q66)·m²n² + Q66(m⁴ + n⁴)
```

### 5.3 Matrices de rigidez de sección: A, B, D `[CLT]`

Se integra analíticamente a través del espesor:

```
// Rigidez extensional A (3×3, índices 11,22,12,16,26,66):
A_ij = Σ_k  Q̄_ij^k · (z_top_k - z_bottom_k)

// Rigidez de acoplamiento membrana-flexión B (3×3):
B_ij = (1/2) · Σ_k  Q̄_ij^k · (z_top_k² - z_bottom_k²)

// Rigidez flexural D (3×3):
D_ij = (1/3) · Σ_k  Q̄_ij^k · (z_top_k³ - z_bottom_k³)
```

> Si el laminado es **simétrico** respecto a la midsurface: `B = 0` (sin acoplamiento).  
> Si es **antisimétrico**: `A₁₆ = A₂₆ = 0` y `D₁₆ = D₂₆ = 0`.

### 5.4 Rigidez de cortante transversal con corrección (Sección 3.6.8)

Para laminados, la distribución de cortante transversal NO es uniforme entre capas.
Abaqus calcula un factor de corrección igualando energías de deformación.

**Paso 1**: Calcular distribución de tensión de cortante transversal por equilibrio:

```
// Para cortante Q_x por unidad de ancho, en la capa i a altura z:
tau_xz(z) en capa i = Q_x · [integral de la distribución de rigidez]
// Usando equilibrio: ∂σ_xx/∂x + ∂τ_xz/∂z = 0
// Con condición de borde: tau_xz = 0 en z = ±h/2

// Para la capa i con coordenada z ∈ [z_i^bot, z_i^top]:
tau_xz^i(z) = -[Q_x / A_xx_eff] ·  Σ_{j=1}^{i-1} [Q̄_11^j · (z_top_j² - z_bot_j²)/2]
              - [Q_x / A_xx_eff] ·  Q̄_11^i · (z² - z_bot_i²) / 2
// (expresión simplificada; el detalle exacto incluye el tensor completo)
```

**Paso 2**: Flexibilidad de cortante de la sección igualando energía:

```
// Energía de cortante real (de la distribución tau):
U_shear_real = (1/2) · Σ_k  ∫_{z_bot_k}^{z_top_k}  tau_xz^k(z)² / G13_k  dz

// Energía de cortante del modelo shell:
U_shear_shell = (1/2) · F_shear · Q_x²

// Igualando: F_shear = U_shear_real · 2 / Q_x²

// Rigidez de cortante efectiva:
K_shear = 1 / F_shear
```

**Paso 3**: La rigidez de cortante transversal efectiva para el elemento:

```
// Rigidez de cortante de la sección (2×2 para las dos direcciones):
D_shear = [[K_s11, K_s12],
           [K_s12, K_s22]]

// K_s11 → corrección en dirección 1
// K_s22 → corrección en dirección 2
// K_s12 → acoplamiento (en general ≠ 0 para laminados no simétricos)
```

> Para el caso isótropo: K_shear = (5/6) · G · h  (factor 5/6 clásico)

### 5.5 Offset de la superficie de referencia (Sección 3.6.8)

Si la superficie de referencia está desplazada de la midsurface por `z₀`:

```
// Relación deformación-posición ajustada:
ε_αβ(z) = ε_αβ^ref + (z - z₀) · κ_αβ

// Matrices de sección con offset:
A_offset = A       // no cambia
B_offset = B - z₀ · A
D_offset = D - 2·z₀ · B + z₀² · A

// Resultantes de sección:
N_αβ = A_offset · ε_αβ^ref + B_offset · κ_αβ
M_αβ = B_offset · ε_αβ^ref + D_offset · κ_αβ

// (z₀ > 0 → offset hacia la superficie superior)
```

---

## 6. Resultantes de Sección

### 6.1 Fuerzas y momentos por integración numérica

Para secciones compuestas Abaqus integra numéricamente a través del espesor
(regla de Simpson o Gauss, según configuración):

```
// En cada punto de Gauss de espesor a coordenada ζ_k, peso w_k:
sigma_αβ(ζ_k) = constitutive_model(ε_αβ + ζ_k · κ_αβ, material^layer(ζ_k))

// Fuerzas de membrana:
N_αβ = Σ_k  sigma_αβ(ζ_k) · w_k · h

// Momentos flectores:
M_αβ = Σ_k  sigma_αβ(ζ_k) · ζ_k · w_k · h

// Cortante transversal (del campo ANS):
Q_α  = D_shear_αβ · γ̄_β   // usando rigidez de sección corregida
```

### 6.2 Principio de trabajo virtual

```
δW_int = ∫_Ω  [N_αβ · δε_αβ^m + M_αβ · δκ_αβ + Q_α · δγ̄_α]  dA
       + δW_hourglass
```

donde `Ω` es la superficie de referencia del elemento.

---

## 7. Control de Hourglass

El elemento S4R con un punto de integración en el plano tiene un modo de energía
cero espurio (hourglass). Se estabiliza con un amortiguamiento pequeño.

### 7.1 Vector de hourglass

```
// Vector de hourglass estándar para elemento cuadrilateral:
h_vec = [1, -1, 1, -1]  // signos alternados en nodos 1-2-3-4

// Ortogonalización respecto a deformación homogénea (en configuración de referencia):
h_orth = h_vec - [(h_vec · u_nodal_bilinear) / (h_vec · h_vec)] · h_vec

// Deformación de hourglass:
z_hg = Σ_I  h_orth^I · u^I   // para cada componente de desplazamiento
```

### 7.2 Rigidez de estabilización

```
// En el plano (membrana):
K_hg_membrane = 0.005 · G · h · A_element   // Abaqus/Standard
K_hg_membrane = 0.050 · G · h · A_element   // Abaqus/Explicit

// Fuerza de hourglass:
F_hg = K_hg_membrane · z_hg

// Trabajo virtual de hourglass:
δW_hg = F_hg · δz_hg
```

### 7.3 Hourglass rotacional

```
// Tensor de hourglass rotacional (derivada cruzada de la normal):
ε_hg_rot = ∂²n̂ / ∂ξ¹∂ξ²  // evaluado en el centro del elemento

// Se actualiza vía cuaterniones al igual que la normal:
ε_hg_rot_new = R(q_inc) · ε_hg_rot_old

// Rigidez rotacional:
K_hg_rot = (stiffness_promedio_de_sección) · A_element · factor_rotacional
```

---

## 8. Integración del Punto de Gauss en el Plano

Para S4R (integración reducida): **1 punto de Gauss** en `(ξ¹, ξ²) = (0, 0)` con peso `w = 4`.

Para S4 (integración completa): **2×2 = 4 puntos de Gauss** con pesos `w = 1` cada uno
en posiciones `±1/√3`.

```
// Peso Gauss para S4:
points_s4r = [(0, 0, 4.0)]
points_s4  = [(-1/√3, -1/√3, 1.0),
              (+1/√3, -1/√3, 1.0),
              (+1/√3, +1/√3, 1.0),
              (-1/√3, +1/√3, 1.0)]
```

Para la **integración de espesor** en sección compuesta Abaqus usa puntos distribuidos
en cada capa (mínimo 3 puntos por capa con la regla de Simpson).

---

## 9. Pipeline de Cómputo — Pseudocódigo

```
// ─────────────────────────────────────────────────────────
// ESTRUCTURAS DE DATOS
// ─────────────────────────────────────────────────────────

struct Layer {
    z_bot, z_top: f64,
    theta: f64,    // ángulo de fibra en grados
    E1, E2: f64,
    nu12: f64,
    G12, G13, G23: f64,
}

struct LaminateSection {
    layers: Vec<Layer>,
    // Matrices de rigidez de sección (calculadas una vez):
    A: Mat3x3,   // extensional
    B: Mat3x3,   // coupling
    D: Mat3x3,   // bending
    K_shear: Mat2x2,  // transverse shear (corrected)
    z_offset: f64,
}

struct NodeState {
    pos: Vec3,       // posición en configuración deformada
    normal: Vec3,    // normal (director)
    disp: Vec3,      // desplazamiento
    rot: Quaternion, // rotación acumulada
}

struct ElementState {
    nodes: [NodeState; 4],
    section: LaminateSection,
    // Estado en punto de integración:
    eps_m: Mat2x2,    // deformación membrana (simétrica)
    kappa: Mat2x2,    // curvatura
    gamma: Vec2,      // cortante transversal
    hg_state: HourglassState,
}

// ─────────────────────────────────────────────────────────
// PASO 1: INICIALIZACIÓN DE SECCIÓN
// (ejecutar una vez por elemento, no en cada iteración)
// ─────────────────────────────────────────────────────────

fn init_section(layers: &[Layer]) -> LaminateSection {
    // 1a. Calcular matrices A, B, D (CLT)
    let mut A, B, D = Mat3x3::zeros()
    for layer in layers {
        let Q_rot = rotate_stiffness(layer.E1, layer.E2, layer.nu12, layer.G12, layer.theta)
        let dz  = layer.z_top - layer.z_bot
        let dz2 = (layer.z_top^2 - layer.z_bot^2) / 2.0
        let dz3 = (layer.z_top^3 - layer.z_bot^3) / 3.0
        A += Q_rot * dz
        B += Q_rot * dz2
        D += Q_rot * dz3
    }

    // 1b. Corrección de cortante transversal por equilibrio
    let K_shear = compute_transverse_shear_stiffness(layers, &A)

    // 1c. Aplicar offset si corresponde
    apply_offset(&mut A, &mut B, &mut D, z_offset)

    LaminateSection { layers, A, B, D, K_shear, z_offset }
}

// ─────────────────────────────────────────────────────────
// PASO 2: CINEMÁTICA POR ITERACIÓN
// ─────────────────────────────────────────────────────────

fn compute_kinematics(elem: &ElementState) -> (StrainMembrane, Curvature, ShearStrain) {
    // 2a. Jacobiano en punto de Gauss (0, 0)
    let J = compute_jacobian(elem.nodes, xi1=0.0, xi2=0.0)
    let J_inv = J.inverse()

    // 2b. Gradiente de deformación incremental
    let F_inc = compute_incremental_deformation_gradient(elem.nodes, J_inv)

    // 2c. Deformación de membrana (log strain via polar decomp.)
    let (R_inc, U_inc) = polar_decompose(F_inc)
    let eps_m = 0.5 * log_of(U_inc^2)   // tensorial

    // 2d. Cambio de curvatura
    let kappa = compute_curvature_change(elem.nodes, J_inv)
    // Incluye gradiente de la actualización de la normal

    // 2e. Cortante transversal ANS (evaluado en midpoints de aristas)
    let gamma_A = evaluate_shear_at_midpoint(elem, xi1=0.0, xi2=-1.0)
    let gamma_C = evaluate_shear_at_midpoint(elem, xi1=0.0, xi2=+1.0)
    let gamma_B = evaluate_shear_at_midpoint(elem, xi1=+1.0, xi2=0.0)
    let gamma_D = evaluate_shear_at_midpoint(elem, xi1=-1.0, xi2=0.0)
    let gamma = interpolate_ANS_shear(gamma_A, gamma_B, gamma_C, gamma_D, xi1=0.0, xi2=0.0)

    (eps_m, kappa, gamma)
}

// ─────────────────────────────────────────────────────────
// PASO 3: FUERZAS INTERNAS
// ─────────────────────────────────────────────────────────

fn compute_internal_forces(elem: &ElementState, eps_m, kappa, gamma) -> NodalForces {
    let sec = &elem.section

    // 3a. Resultantes de sección (para material lineal elástico):
    let N = sec.A * eps_m + sec.B * kappa   // fuerzas de membrana
    let M = sec.B * eps_m + sec.D * kappa   // momentos flectores
    let Q = sec.K_shear * gamma             // cortante transversal

    // Para material no-lineal o plástico: integrar punto a punto a través del espesor
    // evaluando la ley constitutiva en cada capa

    // 3b. Contribuciones al vector de fuerza interna nodal (virtual work)
    let f_int = integrate_virtual_work(elem, N, M, Q)

    // 3c. Sumar hourglass
    let f_hg = compute_hourglass_forces(elem)

    f_int + f_hg
}

// ─────────────────────────────────────────────────────────
// PASO 4: RIGIDEZ TANGENTE (para análisis implícito)
// ─────────────────────────────────────────────────────────

fn compute_tangent_stiffness(elem: &ElementState) -> Mat24x24 {
    // K = K_material + K_geometric + K_hourglass
    let K_mat = integrate_material_stiffness(elem)  // dN/dε, dM/dκ
    let K_geo = integrate_geometric_stiffness(elem)  // tensión inicial × segunda variación
    let K_hg  = compute_hourglass_stiffness(elem)

    K_mat + K_geo + K_hg
}

// ─────────────────────────────────────────────────────────
// PASO 5: ACTUALIZACIÓN DE ESTADO
// ─────────────────────────────────────────────────────────

fn update_state(elem: &mut ElementState, delta_u: NodalDisplacements) {
    for I in 0..4 {
        // Actualizar posición
        elem.nodes[I].pos += delta_u[I].translation

        // Actualizar normal via cuaternión
        let q_inc = quaternion_from_rotation_vector(delta_u[I].rotation)
        elem.nodes[I].normal = q_inc.rotate(elem.nodes[I].normal)
        elem.nodes[I].rot   = q_inc * elem.nodes[I].rot  // composición
    }

    // Actualizar estado de hourglass
    update_hourglass_state(&mut elem.hg_state, delta_u)
}
```

---

## 10. Hipótesis y Limitaciones

| Hipótesis | Implicación de implementación |
|---|---|
| Teoría de primer orden (Mindlin-Reissner) | El cortante transversal es constante en el espesor → corrección por factor k |
| Plano-tensión en cada punto material | σ₃₃ = 0 → el espesor cambia por incompresibilidad efectiva |
| Pequeño cortante transversal | γ_α << 1, permite linealizar la penalización |
| Normal inextensible | El espesor cambia con la deformación de membrana vía Poisson efectivo |
| Continuidad C⁰ | Rotaciones no son continuas entre elementos (compatible con Mindlin) |
| ANS para cortante | Evita locking pero introduce una interpolación no-variacional |

---

## 11. Referencias

- **Abaqus Theory Guide** 2016, §3.6.1, §3.6.5, §3.6.8 — fuente primaria
- MacNeal, R.H. (1978, 1982) — base del método ANS
- Bathe & Dvorkin (1984) — formulación MITC para cortante asumido
- Simo, Fox & Rifai (1989) — teoría de shell finito-strain con cuaterniones
- **Jones, R.M.** — *Mechanics of Composite Materials* — teoría A, B, D `[CLT]`
- **Reddy, J.N.** — *Mechanics of Laminated Composite Plates and Shells* — `[CLT]`
