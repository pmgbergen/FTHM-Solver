import petsc4py
petsc4py.init()
from petsc4py import PETSc

ksp = PETSc.KSP().create()
options = PETSc.Options()
options.setValue('ksp_type', 'gmres')
options.setValue('ksp_rtol', 1e-10)
options.setValue('ksp_gmres_restart', 50)
options.setValue('ksp_max_it', 10)
# options.setValue('ksp_gmres_restart', 50)

ksp.setFromOptions()
# ksp.view()

_ = ksp.destroy()

# options.view()