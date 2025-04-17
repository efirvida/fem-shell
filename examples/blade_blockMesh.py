import os

from fem_shell.cfd_openfoam.blockMesh import BlockMesh
from fem_shell.models.blade.model import Blade

blade_yaml = os.path.join(
    os.getcwd(), "examples", "reference_turbines", "yamls", "BAR0_NREL_1_4_2021.yaml"
)
blade = Blade(blade_yaml=blade_yaml, element_size=0.5)
blade.generate_mesh()
blade.generate_blocks_sections()

bm = BlockMesh(blade.airfoil_list)
bm.write_blockmesh("/home/efirvida/Desktop/dev/fem-shell/examples/test_case")
