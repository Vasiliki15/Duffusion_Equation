from netgen import gui
from ngsolve import *
from netgen.geom2d import SplineGeometry

geo = SplineGeometry()
geo.AddRectangle( (-1, -1), (1, 1),
                 bcs = ("bottom", "right", "top", "left"))
mesh = Mesh( geo.GenerateMesh(maxh=0.25))
Draw(mesh)
fes = H1(mesh, order=3, dirichlet="bottom|top")

u,v = fes.TnT()

time = 0.0
dt = 0.01

gfu = GridFunction(fes)
gfuold = GridFunction(fes)

a = BilinearForm (fes, symmetric=False)
a +=( u*v +dt*3*u**2*grad(u)*grad(v) +dt*u**3*grad(u)*grad(v) - dt*1*v- gfuold*v   ) * dx

from math import pi
gfu = GridFunction(fes)
gfu.Set(sin(2*pi*x))
Draw(gfu,mesh,"u")
SetVisualization (deformation=True)
t = 0

def SolveNonlinearMinProblem(a,gfu,tol=1e-13,maxits=25):
    res = gfu.vec.CreateVector()
    du  = gfu.vec.CreateVector()

    for it in range(maxits):
        print ("Newton iteration {:3}".format(it),end="")
        print ("energy = {:16}".format(a.Energy(gfu.vec)),end="")

        #solve linearized problem:
        a.Apply (gfu.vec, res)
        a.AssembleLinearization (gfu.vec)
        inv = a.mat.Inverse(fes.FreeDofs())
        du.data = inv * res

        #update iteration
        gfu.vec.data -= du

        #stopping criteria
        stopcritval = sqrt(abs(InnerProduct(du,res)))
        print ("<A u",it,", A u",it,">_{-1}^0.5 = ", stopcritval)
        if stopcritval < tol:
            break
        Redraw(blocking=True)

for timestep in range(50):
    gfuold.vec.data = gfu.vec
    SolveNonlinearMinProblem(a,gfu)
    Redraw()
    t += dt
    print("t = ", t)
 
    

