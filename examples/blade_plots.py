import os

from fem_shell.models.blade.mesh import Blade

blade_yaml = os.path.join(
    os.getcwd(), "examples", "reference_turbines", "yamls", "IEA-15-240-RWT_VolturnUS-S.yaml"
)
blade = Blade(blade_yaml=blade_yaml, element_size=0.1)
blade.show_plots()
