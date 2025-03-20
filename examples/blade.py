from os.path import join

from fem_shell.models.blade.mesh import BladeModel

## Define inputs
blade_yaml = join("examples", "blade.yaml")

blade = BladeModel(blade_yaml=blade_yaml, element_size=0.5)
blade.view()
