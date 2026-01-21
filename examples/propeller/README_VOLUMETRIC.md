# Volumetric Remeshing Examples

Este directorio contiene ejemplos de cómo usar las funcionalidades de remallado volumétrico del framework.

## Funcionalidades Nuevas

### `detect_open_boundaries(mesh)`

Detecta si una malla superficial tiene bordes abiertos.

**Parámetros:**
- `mesh`: MeshModel - Malla superficial a verificar

**Retorna:**
- `bool` - True si la malla tiene bordes abiertos, False si está cerrada

**Uso:**
```python
from fem_shell.core.mesh import load_mesh, detect_open_boundaries

mesh = load_mesh("surface_mesh.obj")
has_open_boundaries = detect_open_boundaries(mesh)

if has_open_boundaries:
    print("⚠ La malla tiene bordes abiertos")
else:
    print("✓ La malla está cerrada")
```

### `close_open_boundaries(mesh)`

Cierra los bordes abiertos de una malla superficial rellenando los huecos.

**Parámetros:**
- `mesh`: MeshModel - Malla superficial con bordes abiertos

**Retorna:**
- `MeshModel` - Nueva malla con los bordes cerrados

**Uso:**
```python
from fem_shell.core.mesh import load_mesh, close_open_boundaries

mesh = load_mesh("open_surface.obj")
closed_mesh = close_open_boundaries(mesh)

# Guardar malla cerrada
closed_mesh.write_mesh("closed_surface.stl")
```

**Cómo funciona:**
1. Detecta todos los loops de bordes abiertos
2. Para cada loop, calcula el centroide de los nodos del borde
3. Crea elementos triangulares tipo "fan" desde el centroide hacia cada arista del borde
4. Retorna una nueva malla con los huecos cerrados

### `volumetric_remesh(surface_mesh, target_edge_length=None, algorithm=1, auto_close_boundaries=True)`

Convierte una malla superficial en una malla volumétrica, cerrando automáticamente los bordes abiertos si es necesario.

**Parámetros:**
- `surface_mesh`: MeshModel - Malla superficial de entrada
- `target_edge_length`: float, opcional - Longitud objetivo de las aristas. Si es None, se estima automáticamente.
- `algorithm`: int, opcional - Algoritmo de mallado 3D de Gmsh:
  - 1 = Delaunay (por defecto)
  - 4 = Frontal
  - 7 = MMG3D
  - 10 = HXT
- `auto_close_boundaries`: bool, opcional - Cierra automáticamente bordes abiertos (por defecto: True)

**Retorna:**
- `MeshModel` - Nueva malla con elementos volumétricos (tetraedros)

**Uso:**
```python
from fem_shell.core.mesh import load_mesh, volumetric_remesh

# Cargar malla superficial (puede tener bordes abiertos)
surface_mesh = load_mesh("surface.obj")

# Convertir a malla volumétrica (cierra automáticamente los bordes)
volumetric_mesh = volumetric_remesh(
    surface_mesh, 
    target_edge_length=0.1,
    auto_close_boundaries=True
)

print(f"Nodos: {volumetric_mesh.node_count}")
print(f"Elementos: {volumetric_mesh.elements_count}")

# Guardar malla volumétrica
volumetric_mesh.write_mesh("volumetric_mesh.vtu")
```

## Ejemplos

### 1. `visualize_propeller.py`

Ejemplo que demuestra el flujo completo de trabajo con una malla real:
- Carga una malla superficial de propela (con bordes abiertos)
- Detecta automáticamente los bordes abiertos
- Cierra los bordes automáticamente
- Convierte a malla volumétrica
- Guarda el resultado

**Uso:**
```bash
python visualize_propeller.py
```

**Salida esperada:**
```
Loading mesh from: .../propellerTip.obj
Mesh loaded (meshio format)

Mesh Information (Surface):
  Nodes:    16,785
  Elements: 33,432

Detecting open boundaries...
  ⚠ Open boundaries detected!
  The surface mesh is not closed.

Converting to volumetric mesh...
  Note: This may take several minutes for large meshes...
  ⚠ Open boundaries detected. Attempting to close them...
  Found 1 open boundary loop(s) to close
    Closing loop 1 with 136 boundary nodes...
  ✓ Boundaries closed successfully

Volumetric Mesh Information:
  Nodes:    xxxxx
  Elements: xxxxx

Saving volumetric mesh to: output/propeller_volumetric.vtu
  ✓ Mesh saved successfully!
```

### 2. `volumetric_remesh_example.py`

Ejemplo completo que demuestra el proceso completo:
- Genera una malla superficial cerrada (icosaedro)
- Detecta que está cerrada
- Convierte a malla volumétrica
- Guarda el resultado en `output/sphere_volumetric.vtu`

**Uso:**
```bash
python volumetric_remesh_example.py
```

**Salida esperada:**
```
Generating icosphere surface mesh...

Surface Mesh Information:
  Nodes:    12
  Elements: 20

Detecting open boundaries...
  ✓ Surface mesh is closed.

Converting to volumetric mesh...
  (This may take a moment...)

✓ Volumetric mesh generated successfully!

Volumetric Mesh Information:
  Nodes:    231
  Elements: 715

Element Types:
  tetra: 715

Saving volumetric mesh to: output/sphere_volumetric.vtu
  ✓ Mesh saved successfully!
```

## Notas Importantes

1. **Cierre Automático de Bordes**: La función `volumetric_remesh` ahora puede cerrar automáticamente bordes abiertos usando el parámetro `auto_close_boundaries=True` (valor por defecto).

2. **Algoritmo de Cierre**: Los bordes abiertos se cierran usando triangulación tipo "fan" desde el centroide del loop de borde.

3. **Formatos Soportados**: La función usa internamente el formato STL para el proceso de remallado, que solo soporta elementos triangulares.

4. **Elementos Generados**: Actualmente, la función genera elementos tetraédricos (tetra) para la malla volumétrica.

5. **Rendimiento**: Para mallas grandes (>10,000 nodos), el proceso puede tardar varios minutos. Se recomienda ajustar `target_edge_length` para controlar la densidad de la malla resultante.

6. **Requisitos**:
   - Gmsh debe estar instalado (`pip install gmsh`)
   - La malla superficial debe ser topológicamente válida (sin auto-intersecciones)

## Casos de Uso

- **Análisis por elementos finitos**: Convertir geometrías CAD superficiales a mallas volumétricas para análisis estructural
- **Simulaciones CFD**: Generar dominios internos a partir de superficies cerradas
- **Pre-procesamiento**: Preparar mallas para simulaciones acopladas FSI (Fluid-Structure Interaction)
