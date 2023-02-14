# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 23:35:44 2023

@author: msdaynyba-local
"""

import dolfin
import ufl
from petsc4py import PETSc
import userTKD as userTKD
import numpy as np
import os
import time
import sys


dolfin.PETScOptions.clear()

quadrature_degree = 2
dolfin.parameters["form_compiler"]["quadrature_degree"]=quadrature_degree
dolfin.parameters['form_compiler']['cpp_optimize'] = True
dolfin.parameters['form_compiler']['representation'] = "uflacs"

#%% General Classes and Functions

###############################################################################
class MyNonlinerProblem(dolfin.NonlinearProblem):
    
    """
    Documentation from:
    https://fenics-dolfin.readthedocs.io/en/2017.2.0/apis/api_nls.html?highlight=nonlinearproblem#_CPPv2N6dolfin16NonlinearProblemE
    """
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form   = F
        self.bcs = bcs
        dolfin.NonlinearProblem.__init__(self)

    def tangent(self, A, x):
        dolfin.assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)
            
    def residual(self, b, x):
        dolfin.assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)
            
    def setFields(self):
        pass
    
class TKD_NonlinerProblem_poroelastic(dolfin.NonlinearProblem):
    
    """
    Documentation from:
    https://fenics-dolfin.readthedocs.io/en/2017.2.0/apis/api_nls.html?
    highlight=nonlinearproblem#_CPPv2N6dolfin16NonlinearProblemE
    """
    def __init__(self, rhs, bcs,w0,dt, w=None, dw=None, delta_w=None, 
                 dX=None, mesh=None, 
                 degreeF=1, degreeP=1, quadrature=2,
                 prestress=dolfin.Constant(0), **params):
        
        self.w0=w0
        (self.u0, self.p0) = dolfin.split(self.w0)

        self.dt=dt


        # FOR WEAK FORM
        self.rhs = rhs
        self.bcs = bcs
        
        # my own variables
        self.w = w               # solution field
        self.dw = dw             # trial field
        self.delta_w = delta_w   # virtual field
        
        (self.du, self.dp) = dolfin.split(self.dw)
        (self.delta_u, self.delta_p) = dolfin.split(self.delta_w)
        (self.u, self.p) = dolfin.split(self.w)
                
        # MESH
        self.dX = dX             # differential volume (dolfin.dx)
        self.mesh = mesh
        
        # PARAMETERS FOR CONSTITUTIVE RELATION
        self.I = dolfin.Identity(3)
        self.mu    = params['mu']
        self.fo    = params['fo']
        self.d     = params['d']
        self.alpha = params['alpha']
        self.K = params['K']
        
        # EULER-LAGRANGE DEFORMATION TENSOR
        self.grad_phi = dolfin.grad(self.u) + self.I
        self.grad_phi_inv  = dolfin.inv(self.grad_phi)
        self.grad_phi_invt = self.grad_phi_inv.T
        self.J = dolfin.det(self.grad_phi)  
        self.delta_E  = dolfin.sym(self.grad_phi.T * dolfin.grad(self.delta_u))
        self.DE       = dolfin.sym(self.grad_phi.T * dolfin.grad(self.du))
        self.Ddelta_E = dolfin.sym((dolfin.grad(self.du)).T*dolfin.grad(self.delta_u))
        # VOIGT NOTATION
        self.S_voigt = None

        self.delta_E_voigt = dolfin.as_vector((self.delta_E[0,0], 
                                               self.delta_E[1,1], 
                                               self.delta_E[2,2],
                                               self.delta_E[0,1] + self.delta_E[1,0],
                                               self.delta_E[1,2] + self.delta_E[2,1],
                                               self.delta_E[0,2] + self.delta_E[2,0]))

        self.DE_voigt = dolfin.as_vector((self.DE[0,0], 
                                          self.DE[1,1], 
                                          self.DE[2,2],
                                          self.DE[0,1] + self.DE[1,0],
                                          self.DE[1,2] + self.DE[2,1],
                                          self.DE[0,2] + self.DE[2,0]))
        
        # INITIAL STRESS STATE
        self.prestress = prestress
        # sigma_initial = dolfin.Identity(3) * initial_hydrostatic_stress
        # self.S_initial = self.J * self.grad_phi_inv * sigma_initial * self.grad_phi_invt
    
        # FOR USER ROUTINE
        # local deformation gradient and pressure
        PF_elem = dolfin.TensorElement("DG", mesh.ufl_cell(), degreeF)
        Pplocal_elem = dolfin.FiniteElement("DG", mesh.ufl_cell(), degreeP)
        TH = dolfin.MixedElement([PF_elem, Pplocal_elem]) 
        Wloc  = dolfin.FunctionSpace(mesh, TH)
        self.wloc       = dolfin.Function(Wloc)  


        # field assigner
        PF = dolfin.TensorFunctionSpace(mesh, 'DG', degreeP)
        Pplocal = dolfin.FunctionSpace(mesh, 'DG', degreeF)
        self.assigner = dolfin.FunctionAssigner(Wloc, [PF, Pplocal])

        self.dF      = dolfin.TrialFunction(PF)
        self.delta_F = dolfin.TestFunction(PF)
        self.F       = dolfin.Function(PF)      
        
        self.dplocal      = dolfin.TrialFunction(Pplocal)
        self.delta_plocal = dolfin.TestFunction(Pplocal)
        self.plocal       = dolfin.Function(Pplocal)      
        
        
        # element
        QE1 = dolfin.TensorElement(family="Quadrature", cell=self.mesh.ufl_cell(),
                                       degree=quadrature, quad_scheme="default")
        self.QE = dolfin.MixedElement([QE1, QE1, QE1]) 
        
        # constitutive relation
        self.userConstitutive = dolfin.compile_cpp_code(userTKD.constitutive_relation)
        self.userCRdata = None
        
        # inheritance
        dolfin.NonlinearProblem.__init__(self)

    #########################################################################
    def tangent(self, A, x):
        
        c = self.userCRdata
        idx = 6
        Celast_voigt = dolfin.as_matrix(( (c[idx+0], c[idx+1],  c[idx+2],  c[idx+3],  c[idx+4],  c[idx+5]),
                                           (c[idx+1], c[idx+6],  c[idx+7],  c[idx+8],  c[idx+9],  c[idx+10]),
                                           (c[idx+2], c[idx+7],  c[idx+11], c[idx+12], c[idx+13], c[idx+14]),
                                           (c[idx+3], c[idx+8],  c[idx+12], c[idx+15], c[idx+16], c[idx+17]),
                                           (c[idx+4], c[idx+9],  c[idx+13], c[idx+16], c[idx+18], c[idx+19]),
                                           (c[idx+5], c[idx+10], c[idx+14], c[idx+17], c[idx+19], c[idx+20]) ) )

        #u_u
        Duu1 = dolfin.inner(self.delta_E_voigt, dolfin.dot(Celast_voigt, self.DE_voigt))*self.dX + \
                dolfin.inner(self.Ddelta_E, self.S)*self.dX # derivada respecto a du, del #primer término de principio de trabajos virtuales (esfuerzo efectivo)
        # Duu1_initial = dolfin.derivative(self.a_uinitial, self.u, self.du)
        Duu2 = dolfin.derivative(self.a_up, self.u, self.du) #derivada respecto a du, del segundo término de principio de trabajos virtuales (esfuerzo en fluido) 
        
        #u_p
        # Dup = dolfin.derivative(self.a_uu + self.a_uinitial + self.a_up, self.p, self.dp)
        Dup = dolfin.derivative(self.a_uu + self.a_up, self.p, self.dp) #derivada respecto a dp, del primer y segundo término de principio de trabajos virtuales
        
        # p_u & p_p
        Dpu = dolfin.derivative(self.a_p, self.u, self.du) #derivada respecto a du de la conservación de la masa
        Dpp = dolfin.derivative(self.a_p, self.p, self.dp) #derivada respecto a dp de la conservación de la masa
        
        # external forces
        DL  = dolfin.derivative(self.rhs, self.w, self.dw) # derivada de presión pleural

        # DG = (Duu1 + Duu1_initial + Duu2 + Dup) + (Dpu + Dpp) - DL        
        DG = (Duu1 + Duu2 + Dup) + (Dpu + Dpp) - DL 
      
        # To compare with exact solution
        # DG = dolfin.derivative(self.G, self.w, self.dw)
        
        dolfin.assemble(DG, tensor=A)
        for bc in self.bcs:
            bc.apply(A)
            
    #########################################################################
    def residual(self, b, x):
        
        # update S and calculate new left hand side
        c = self.userCRdata
        self.S = dolfin.as_tensor( ( (c[0], c[1], c[2]), (c[1], c[3], c[4]), (c[2], c[4], c[5]) ) )        
        
        phi = self.J - (1.-self.fo)
        Z = self.J * self.grad_phi_inv * self.K * self.grad_phi_inv.T * dolfin.grad(self.p) #ley de darcy
        self.F2aux1=self.J*dolfin.tr( (dolfin.grad(self.u)-dolfin.grad(self.u0)) *self.grad_phi_inv ) #Variable auxiliar para conservacion de la masa dinamica

        self.a_uu = dolfin.inner(self.delta_E, self.S)*self.dX #primer término de principio de trabajos virtuales (esfuerzo efectivo)        
        self.a_up = -self.J*self.p  * dolfin.inner(dolfin.grad(self.delta_u),  self.grad_phi_invt)*self.dX    #segundo término de principio de trabajos virtuales (esfuerzo en fluido)    

        
        
        if self.dt.beta==-1:  # prestress
          print('prestress phase:',self.dt.beta==-1)
          print('prestress value',float(self.prestress))
          self.a_p  = dolfin.inner(dolfin.grad(self.delta_p), Z)*self.dX  #conservación de masa - estatica
        else:
          self.a_p  = ( (dolfin.inner(self.F2aux1, self.delta_p))*self.dX)+self.dt*dolfin.inner(dolfin.grad(self.delta_p), Z)*self.dX  #conservación de masa - dinamica


        # G = (self.a_uu + self.a_uinitial + self.a_up) + self.a_p - self.rhs
        G = (self.a_uu + self.a_up) + self.a_p - self.rhs #rhsl tiene presi\'on pleural
        
        #u0_array = u0.vector().get_local()
        #print(u0_array)
        """
        self.ua=self.u-self.u0
        Vfs = dolfin.VectorFunctionSpace(mesh, 'CG', 1)
        ua_proj = dolfin.project(self.ua, Vfs)
        ua_values  = ua_proj.vector().get_local()#array()
        print(ua_values)

        Efs = dolfin.FunctionSpace(mesh, 'CG', 1)
        p_proj = dolfin.project(self.p, Efs)
        p_values  = p_proj.vector().get_local()#array()
        print(p_values)
        #print(u0_values)
        """
        dolfin.assemble(G, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)
     
    #########################################################################
    def setFields(self):

        # solve local problem
        a_local = dolfin.inner( self.dF, self.delta_F ) * self.dX
        L_local = dolfin.inner( self.grad_phi, self.delta_F) * self.dX
        ls = dolfin.LocalSolver(a_local, L_local)
        ls.solve_global_rhs(self.F)
        
        a_local = dolfin.inner( self.dplocal, self.delta_plocal ) * self.dX
        L_local = dolfin.inner( self.p, self.delta_plocal) * self.dX
        ls = dolfin.LocalSolver(a_local, L_local)
        ls.solve_global_rhs(self.plocal)
        
        # mixed element
        self.assigner.assign(self.wloc, [self.F, self.plocal])

        # USER CONSTITUTIVE
        
        self.userCRdata = dolfin.CompiledExpression(self.userConstitutive.TKD(self.wloc.cpp_object()), 
                                                    mu=self.mu,
                                                    fo=self.fo,
                                                    d=self.d,
                                                    alpha=self.alpha,
                                                    prestress = self.prestress,
                                                    element=self.QE)




###############################################################################
class MyNewtonSolver(dolfin.NewtonSolver):
    
    """
    Documentation from: 
    https://fenics-dolfin.readthedocs.io/en/2017.2.0/apis/api_nls.html#id2
    """
    
    ##############################################################################
    def __init__(self, mesh):
        
        factory = dolfin.PETScFactory.instance()  # The factory
        comm = mesh.mpi_comm()                    # MPI communicator
        linear_solver_type = dolfin.PETScLUSolver()
    
        # initiate fenics Newton Solver object
        dolfin.NewtonSolver.__init__(self, comm, linear_solver_type, factory)
        
        # attributes
        self.newton_iteration     = 0
        self.krylov_iterations    = 0
        self.relaxation_parameter = 1.
        self.residual             = 0.
        self.solver = self.linear_solver()
        self.matA   = factory.create_matrix(comm)
        self.matP   = factory.create_matrix(comm)
        self.dx     = factory.create_vector(comm)
        self.b      = factory.create_vector(comm)
        self.mpi_comm = comm
        self.factory  = factory

    
    ##############################################################################
    def solve(self, myproblem, x):
        
        # Extract parameters for parent NewtonRapson class
        convergence_criterion = self.parameters['convergence_criterion']
        maxiter =               self.parameters['maximum_iterations']
        
        # Reset iteration counts
        self.newton_iteration  = 0
        self.krylov_iterations = 0
        
        # Compute F(u)
        myproblem.form(self.matA, self.matP, self.b, x)
        myproblem.setFields()
        myproblem.residual(self.b, x)
        
        # Check convergence
        newton_converged = False
        if convergence_criterion == 'residual':
            newton_converged = self.converged(self.b, myproblem, 0)
        elif convergence_criterion == 'incremental':
            newton_converged = False
        else:
            raise ValueError("The convergence criterion %s is unknown, known criteria are 'residual' or 'incremental'")
        
        
        # Start iterations
        while (not newton_converged and  self.newton_iteration <= maxiter):
            
            self.matA = self.factory.create_matrix(self.mpi_comm)
            
            # compute Jacobian
            myproblem.tangent(self.matA, x)
            
            # Setup (linear) solver (including set operators)
            self.solver_setup(self.matA, self.matP, myproblem, self.newton_iteration);
            
            # Perform linear solve and update total number of Krylov
            # iterations
            if not self.dx.empty():     self.dx.zero()
            self.krylov_iterations += self.solver.solve(self.dx, self.b);
                
            # Update solution
            self.update_solution(x, self.dx, self.relaxation_parameter, myproblem, self.newton_iteration);
            self.newton_iteration += 1

            # compute deformation gradients            
            myproblem.setFields()

            # Compute F
            myproblem.form(self.matA, self.matP, self.b, x)
            myproblem.residual(self.b, x);            
            
            # Test for convergence
            if convergence_criterion == "residual":
                newton_converged = self.converged(self.b, myproblem, self.newton_iteration);
            elif convergence_criterion == "incremental":
                # Subtract 1 to make sure that the initial residual0 is properly set.
                newton_converged = self.converged(self.dx, myproblem, self.newton_iteration - 1);
            else:
                raise ValueError("The convergence criterion %s is unknown, known criteria are 'residual' or 'incremental'")
            
            
        if newton_converged:
            if self.mpi_comm.rank == 0:
              dolfin.info("Newton solver finished in %d" % self.newton_iteration + 
                     " iterations and %d" % self.krylov_iterations + " linear solver iterations.")
        else:
            error_on_nonconvergence = self.parameters["error_on_nonconvergence"]
            if error_on_nonconvergence:
                if self.newton_iteration == maxiter:
                    raise ValueError("NewtonSolver.cpp",
                         "solve nonlinear system with NewtonSolver",
                         "Newton solver did not converge because maximum number of iterations reached")
                  
                else:
                    raise ValueError("NewtonSolver.cpp",
                                 "solve nonlinear system with NewtonSolver",
                                 "Newton solver did not converge")
            else:
                print("Newton solver did not converge.")
        
        return self.newton_iteration, newton_converged
    
    
    ##############################################################################
    def update_solution(self, x, dx, relaxation_parameter, problem, iteration):
        if relaxation_parameter == 1.0:
            x -= dx
        else:
            x.axpy(-relaxation_parameter, dx)


def give_times_fluxes_deltas_VCV(Nciclos, vol_step,area,Tsyr,Texp,Tpausa,Tinsp):
    q_step=vol_step/(area*Tsyr)
    #########TIEMPOS, FLUJOS Y PASOS#####################
    for ciclo in np.arange(1,Nciclos+1):

        duration_step=Tsyr+Tpausa+Texp #duracion ciclo inspiracion-expiracion
        t0=duration_step*(ciclo-1)
        t1=0.001+t0
        t2=Tsyr+t0
        t3=Tsyr+t0+0.001  #tiempo cuando finaliza la insuflacion, por lo que comienza la pausa
        t4=Tsyr+Tpausa+t0  #tiempo que finaliza la pausa y comienza presion =0 , cambio de BC
        t5=t4+Tpausa #tiempo auxiliar para detertar peak negativo
        t6=duration_step+t0 #fin de ciclo 
        
        print(t0,t1,t2,t3,t4,t5,t6)
    
        if ciclo<=Nciclos/2:
            factor=1
        else:
            factor=+1#-1
        q0=0
        q1=-q_step*factor
        q2=-q_step*factor
        q3=0
        q4=0
        q5=0
        q6=0

#        n0=3 
#        n1=10
#        n2= 3
#        n3=10
#        n4= 10
#        n5=15
        
        n0=3 
        n1=5
        n2= 3
        n3=5
        n4= 5
        n5=5
        #0.0 0.001 0.99 1.0 1.5 4.5
        if ciclo==Nciclos:    
            end=True
        else:
            end=False
            
        times0=np.linspace(t0,t1,n0,endpoint=False)
        times1=np.linspace(t1,t2,n1,endpoint=False)
        times2=np.linspace(t2,t3,n2,endpoint=False)
        times3=np.linspace(t3,t4,n3,endpoint=False)
        times4=np.linspace(t4,t5,n4,endpoint=False)
        times5=np.linspace(t5,t6,n5,endpoint=end)
        timesaux=np.concatenate((times0,times1,times2,times3,times4,times5))
        

        qs0=np.linspace(q0,q1,n0,endpoint=False)
        qs1=np.linspace(q1,q2,n1,endpoint=False)
        qs2=np.linspace(q2,q3,n2,endpoint=False)
        qs3=np.linspace(q3,q4,n3,endpoint=False)
        qs4=np.linspace(q3,q4,n4,endpoint=False)
        qs5=np.linspace(q4,q5,n5,endpoint=end)
        qsaux=np.concatenate((qs0,qs1,qs2,qs3,qs4,qs5))
        
        if ciclo==1:
            times=timesaux
            qs=qsaux
        else:
            times=np.concatenate((times,timesaux))
            qs=np.concatenate((qs,qsaux))
    times=times[1:]
    qs=qs[1:]
    
    dts=[]
    for i in np.arange(len(times)):
        if i==0:
            dts.append(times[i])
        else:
            dts.append(times[i]-times[i-1]) 

    return times,qs,dts


def give_VTKfiles(lung,mesh,signal_list,h5_file,NLproblem,NLproblemB,permeability,ventilation_params,caso,u,p,dx,i,t):
    times,fluxes,presionestodas,Jacob=signal_list
    Tsyr,Texp,Tpausa,vol_step,Nciclo=ventilation_params
    Tinsp=Tsyr+Tpausa
    duration_step=Tsyr+Tpausa+Texp

    if round(t,5)==Tsyr or round(t,5)==(Tsyr+Tpausa) or round(t,5)==(Tsyr+Tpausa+0.5*Texp) or round(t,8)==1.0e-03:
        Efs = dolfin.FunctionSpace(mesh, 'CG', 1)
        Vfs = dolfin.VectorFunctionSpace(mesh, 'CG', 1)
        I  = dolfin.Identity(3)
        grad_phi = dolfin.grad(u) + I
        J  = dolfin.det(grad_phi)    
        K = dolfin.Identity(3)*permeability  
        # get fields
        print('Projecting solution fields...')
        dolfin.File(lung+'/'+'Results/'+'VTKfiles/'+caso+"/Pressure"+"time"+str(t)+".pvd")<<(p,t)
        dolfin.File(lung+'/'+'Results/'+'VTKfiles/'+caso+"/Displacement"+"time"+str(t)+".pvd")<<(u,t)
        

        # recover jacobian projections
        J_proj = dolfin.project(J, Efs, solver_type='mumps')
        J_proj.rename('Jacobian','Jacobian')
        h5_file.write(J_proj, 'J_proj', float(i))  #i-1?
        dolfin.File(lung+'/'+'Results/'+'VTKfiles/'+caso+"/Jacobian"+"time"+str(t)+".pvd")<<J_proj


        # Cauchy stress
        if t<=Tinsp: #FIN  DE INSPIRACION (INSUFLACION + PAUSA)
            S = NLproblem.S
        elif Tinsp<t<=duration_step:
            S = NLproblemB.S
        elif duration_step<t<duration_step+Tinsp:
          S = NLproblem.S
        else:
          S = NLproblemB.S

        sigma = 1./J * grad_phi * S * grad_phi.T
        # hydrostatic pressure and Von Mises stress (https://en.wikipedia.org/wiki/Von_Mises_yield_criterion)
        sigma_hydro = 1./3.*dolfin.tr(sigma)
        sigma_VM = dolfin.sqrt(0.5*( (sigma[0,0]-sigma[1,1])**2 +  (sigma[1,1]-sigma[2,2])**2 + \
                                      (sigma[2,2]-sigma[0,0])**2 + \
                                      6.*(sigma[0,1]**2 + sigma[1,2]**2 + sigma[2,0]**2) ))
        
        sigma_hydro_proj = dolfin.project(sigma_hydro, Efs, solver_type='mumps')
        sigma_hydro_proj.rename('sigma_hydro','sigma_hydro')
        h5_file.write(sigma_hydro_proj, 'sigma_hydro_proj', float(i))  #i-1?
        dolfin.File(lung+'/'+'Results/'+'VTKfiles/'+caso+"/HYD_ti"+"time"+str(t)+".pvd")<<sigma_hydro_proj


        sigma_VM_proj = dolfin.project(sigma_VM, Efs, solver_type='mumps')
        sigma_VM_proj.rename('sigma_VM','sigma_VM')
        h5_file.write(sigma_VM_proj, 'sigma_VM_proj', float(i))  #i-1?
        dolfin.File(lung+'/'+'Results/'+'VTKfiles/'+caso+"/VM_ti"+"time"+str(t)+".pvd")<<sigma_VM_proj
        
        #Darcy flux
        Qflux=J*dolfin.inv(grad_phi)*K*dolfin.inv(grad_phi).T*(-dolfin.grad(p))
        Qflux_proj=dolfin.project(Qflux,Vfs,solver_type='mumps')
        Qflux_proj.rename('Qflux','Qflux')
        h5_file.write(Qflux_proj, 'Qflux_pro', float(i))
        dolfin.File(lung+'/'+'Results/'+'VTKfiles/'+caso+"/Qflux"+"time"+str(t)+".pvd")<<Qflux_proj            
    
    return



def create_folders(lung,caso):
    # Create Results directory
    dirName = lung+'/'+'Results/'
    try:
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

    # Create Results directory
    dirName = lung+'/'+'Results/Signals'
    try:
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
        
        
    # Create directory for 3D fields
    dirNamefields = lung+'/'+'Results/VTKfiles'+caso
    try:
        os.mkdir(dirNamefields)
        print("Directory " , dirNamefields ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirNamefields ,  " already exists")        
    return





import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model




def give_tpqv(folder,name):
  ad=102.95
  ai=117.08
  at=ad+ai
  flujos=np.load('Right/Results/Signals'+folder+'/'+name+'fluxes.npy')*60+np.load('Right/Results/Signals'+folder+'/'+name+'fluxes.npy')*60
  presiones=(ad*np.load('Right/Results/Signals'+folder+'/'+name+'presionestodas.npy')+ai*np.load('Right/Results/Signals'+folder+'/'+name+'presionestodas.npy'))/at
  volumenes=np.load('Right/Results/Signals'+folder+'/'+name+'volumenes.npy')+np.load('Right/Results/Signals'+folder+'/'+name+'volumenes.npy')
  volumenes=volumenes-volumenes[0]
  tiempos=np.load('Right/Results/Signals'+folder+'/'+name+'tiempos.npy')

  flujos=np.concatenate((np.array([0]),np.asarray(flujos)))
  presiones=np.concatenate((np.array([0]),np.asarray(presiones)))
  tiempos=np.concatenate((np.array([0]),np.asarray(tiempos)))
  volumenes=np.concatenate((np.array([0]),np.asarray(volumenes)))

  return tiempos,presiones,flujos,volumenes 


#DESCOMENTAR PARA LETRA LATEX
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    "font.serif": ["Palatino"]})


def plotVCV():
    print('Sensitibity analysis for f0')
    folder1='/fo0.55_mu10.33s00'
    folder2='/fo0.66_mu10.33s00'
    folder3='/fo0.69_mu10.33s00'
    folder4='/fo0.72_mu10.33s00'
    folder5='/fo0.83_mu10.33s00'
    
    name='multi'
    color25='black'
    alpha25=0.2
    style25='-'
    
    color5='black'
    alpha5=0.3
    style5='-'
    
    colornormal='black'
    alphanormal=1
    stylenormal='--'
    
    tmenos25,pmenos25, qmenos25,vmenos25=give_tpqv(folder1,name)
    tmas25,pmas25, qmas25,vmas25=give_tpqv(folder5,name)
    
    tmenos5,pmenos5, qmenos5,vmenos5=give_tpqv(folder2,name)
    tmas5,pmas5, qmas5,vmas5=give_tpqv(folder4,name)
    
    tnormal,pnormal, qnormal,vnormal=give_tpqv(folder3,name)
    
    
    fig, axs = plt.subplots(nrows=3, ncols=1)
    ax1=axs[0]
    ax2=axs[1]
    ax3=axs[2]
    fig.set_size_inches(5, 7)
    
    position='upper right'
    limx=max(tmenos25)
    limxmin=-0.02
    si=15
    minor_ticks_left=np.linspace(0,6,4)
    
    
    #ax1.fill_between(tiempos, presiones*10.2, y2=0,color='k',alpha=0.1)
    ax1.fill_between(tmenos25, y1=0, y2=pmenos25*10.2,color=color5,alpha=alpha5+alpha25,label='±$5\%$ ($f_0$)')
    ax1.fill_between(tmenos25, y1=0, y2=pmenos25*10.2,color=color25,alpha=alpha25,label='±$20\%$ ($f_0$)')
    ax1.fill_between(tmenos25, y1=0, y2=pmenos25*10.2,color='w')
    
    
    ax1.fill_between(tmenos25, y1=pmenos25*10.2, y2=pmas25*10.2,color=color25,alpha=alpha25)
    ax1.plot(tmenos25,pmenos25*10.2,color=color25,linestyle=style25,alpha=alpha25)
    ax1.plot(tmas25,pmas25*10.2,color=color25,linestyle=style25,alpha=alpha25)
    ax1.fill_between(tmenos5, y1=pmenos5*10.2, y2=pmas5*10.2,color=color5,alpha=alpha5)
    ax1.plot(tmenos5,pmenos5*10.2,color=color5,linestyle=style5,alpha=alpha5)
    ax1.plot(tmas5,pmas5*10.2,color=color5,linestyle=style5,alpha=alpha5)
    ax1.plot(tnormal,pnormal*10.2,color=colornormal,linestyle=stylenormal,alpha=alphanormal)
    
    
    
    
    ax1.set_ylabel(r'Aw. pressure [cm H$_2$O]',size=si)
    ax1.set_xlim(limxmin,limx)
    ax1.set_yticks([0, 4,8])
    ax1.set_yticklabels([r"$0$",r"$4$",r"$8$"], color="k", size=si)
    ax1.tick_params(bottom=False, labelbottom=False)
    ax1.set_ylim(0-0.05,8+0.5)
    labelx = -0.15  # axes coords
    ax1.yaxis.set_label_coords(labelx, 0.5)
    ax1.legend(bbox_to_anchor=(0,1.1,1,0.2),loc='upper left',mode='expand',ncol=6,shadow=True,prop=dict(size=si*0.9)) #bbox_to_anchor=(x0,y0, width, height)
    
    
    
    minor_ticks_left=np.linspace(-80,80,3)
    ax2.fill_between(tmenos25, y1=qmenos25, y2=qmas25,color=color25,alpha=alpha25,label='±25\%')
    ax2.plot(tmenos25,qmenos25,color=color25,linestyle=style25,alpha=alpha25)
    ax2.plot(tmas25,qmas25,color=color25,linestyle=style25,alpha=alpha25)
    ax2.fill_between(tmenos5, y1=qmenos5, y2=qmas5,color=color5,alpha=alpha5,label='±5\%')
    ax2.plot(tmenos5,qmenos5,color=color5,linestyle=style5,alpha=alpha5)
    ax2.plot(tmas5,qmas5,color=color5,linestyle=style5,alpha=alpha5)
    ax2.plot(tnormal,qnormal,color=colornormal,linestyle=stylenormal,alpha=alphanormal)
    
    
    
    #ax2.set_ylim(-100,100)
    ax2.set_ylabel(r'Flow [L/min]',size=si)
    ax2.axhline(0,color='black',linestyle="-.",lw=0.5)
    ax2.set_xlim(limxmin,limx)
    labelx = -0.15  # axes coords
    ax2.yaxis.set_label_coords(labelx, 0.5)
    ax2.set_yticks([-80, 0, 40])
    ax2.set_yticklabels([r"$-80$", r"$0$", r"$40$"], color="k", size=si)
    ax2.set_ylim(-85,50)
    ax2.tick_params(bottom=False, labelbottom=False)
    
    
    
    minor_ticks_left=np.linspace(0,0.8,3)
    ax3.fill_between(tmenos25, y1=vmenos25, y2=vmas25,color=color25,alpha=alpha25,label='±25\%')
    ax3.plot(tmenos25,vmenos25, color=color25,linestyle=style25,alpha=alpha25)
    ax3.plot(tmas25,vmas25, color=color25,linestyle=style25,alpha=alpha25)
    
    ax3.fill_between(tmenos5, y1=vmenos5, y2=vmas5,color=color5,alpha=alpha5,label='±5\%')
    ax3.plot(tmenos5,vmenos5, color=color5,linestyle=style5,alpha=alpha5)
    ax3.plot(tmas5,vmas5, color=color5,linestyle=style5,alpha=alpha5)
    
    ax3.plot(tnormal,vnormal,color=colornormal,linestyle=stylenormal,alpha=alphanormal)
    
    
    ax3.grid(which='minor')
    ax3.set_ylabel(r'Volume [L]',size=si)
    ax3.set_xlabel(r'Time [s]',size=si)
    ax3.set_xlim(limxmin,limx)
    ax3.set_yticks([0,0.2, 0.4])
    ax3.set_yticklabels(["0",r"$0.2$", r"$0.4$"], color="k", size=si)
    labelx = -0.15  # axes coords
    ax3.yaxis.set_label_coords(labelx, 0.5)
    ax3.set_xticks([0, 1,2,3])
    ax3.set_xticklabels([r"$0$", r"$1$", r"$2$", r"$3$"], color="k", size=si)
    ax3.set_ylim(0,0.52)
    plt.tight_layout()     
    plt.show()
    plt.savefig('SAfo_vcv.pdf')
    
    
    
    
    #%GRAFICOS ANALISIS DE SENSIBILIDAD PARA \mu
    print('---------------------------------------------------------------')
    print('Sensitibity analysis for \mu')

    folder1='/fo0.69_mu8.26s00'
    folder2='/fo0.69_mu9.81s00'
    folder3='/fo0.69_mu10.33s00'
    folder4='/fo0.69_mu10.85s00'
    folder5='/fo0.69_mu12.4s00'
    
    
    
    name='multi'
    color25='black'
    alpha25=0.2
    style25='-'
    
    color5='black'
    alpha5=0.3
    style5='-'
    
    colornormal='black'
    alphanormal=1
    stylenormal='--'
    
    tmenos25,pmenos25, qmenos25,vmenos25=give_tpqv(folder1,name)
    tmas25,pmas25, qmas25,vmas25=give_tpqv(folder5,name)
    
    tmenos5,pmenos5, qmenos5,vmenos5=give_tpqv(folder2,name)
    tmas5,pmas5, qmas5,vmas5=give_tpqv(folder4,name)
    
    tnormal,pnormal, qnormal,vnormal=give_tpqv(folder3,name)
    
    
    fig, axs = plt.subplots(nrows=3, ncols=1)
    ax1=axs[0]
    ax2=axs[1]
    ax3=axs[2]
    fig.set_size_inches(5, 7)
    
    position='upper right'
    limx=max(tmenos25)
    minor_ticks_left=np.linspace(0,6,4)
    
    
    #ax1.fill_between(tiempos, presiones*10.2, y2=0,color='k',alpha=0.1)
    ax1.fill_between(tmenos25, y1=0, y2=pmenos25*10.2,color=color5,alpha=alpha5+alpha25,label='±$5\%$ ($\mu$)')
    ax1.fill_between(tmenos25, y1=0, y2=pmenos25*10.2,color=color25,alpha=alpha25,label='±$20\%$ ($\mu$)')
    ax1.fill_between(tmenos25, y1=0, y2=pmenos25*10.2,color='w')
    
    
    ax1.fill_between(tmenos25, y1=pmenos25*10.2, y2=pmas25*10.2,color=color25,alpha=alpha25)
    ax1.plot(tmenos25,pmenos25*10.2,color=color25,linestyle=style25,alpha=alpha25)
    ax1.plot(tmas25,pmas25*10.2,color=color25,linestyle=style25,alpha=alpha25)
    ax1.fill_between(tmenos5, y1=pmenos5*10.2, y2=pmas5*10.2,color=color5,alpha=alpha5)
    
    ax1.plot(tmenos5,pmenos5*10.2,color=color5,linestyle=style5,alpha=alpha5)
    ax1.plot(tmas5,pmas5*10.2,color=color5,linestyle=style5,alpha=alpha5)
    ax1.plot(tnormal,pnormal*10.2,color=colornormal,linestyle=stylenormal,alpha=alphanormal)
    
    
    
    
    ax1.set_ylabel(r'Aw. pressure [cm H$_2$O]',size=si)
    ax1.set_xlim(limxmin,limx)
    ax1.set_yticks([0, 4,8])
    ax1.set_yticklabels([r"$0$",r"$4$",r"$8$"], color="k", size=si)
    ax1.tick_params(bottom=False, labelbottom=False)
    ax1.set_ylim(0-0.05,8+0.5)
    labelx = -0.15  # axes coords
    ax1.yaxis.set_label_coords(labelx, 0.5)
    ax1.legend(bbox_to_anchor=(0,1.1,1,0.2),loc='upper left',mode='expand',ncol=6,shadow=True,prop=dict(size=si*0.9)) #bbox_to_anchor=(x0,y0, width, height)
    
    
    
    minor_ticks_left=np.linspace(-80,80,3)
    ax2.fill_between(tmenos25, y1=qmenos25, y2=qmas25,color=color25,alpha=alpha25,label='±25\%')
    ax2.plot(tmenos25,qmenos25,color=color25,linestyle=style25,alpha=alpha25)
    ax2.plot(tmas25,qmas25,color=color25,linestyle=style25,alpha=alpha25)
    ax2.fill_between(tmenos5, y1=qmenos5, y2=qmas5,color=color5,alpha=alpha5,label='±5\%')
    ax2.plot(tmenos5,qmenos5,color=color5,linestyle=style5,alpha=alpha5)
    ax2.plot(tmas5,qmas5,color=color5,linestyle=style5,alpha=alpha5)
    ax2.plot(tnormal,qnormal,color=colornormal,linestyle=stylenormal,alpha=alphanormal)
    
    
    
    #ax2.set_ylim(-100,100)
    ax2.set_ylabel(r'Flow [L/min]',size=si)
    ax2.axhline(0,color='black',linestyle="-.",lw=0.5)
    ax2.set_xlim(limxmin,limx)
    labelx = -0.15  # axes coords
    ax2.yaxis.set_label_coords(labelx, 0.5)
    ax2.set_yticks([-80, 0, 40])
    ax2.set_yticklabels([r"$-80$", r"$0$", r"$40$"], color="k", size=si)
    ax2.set_ylim(-85,50)
    ax2.tick_params(bottom=False, labelbottom=False)
    
    
    
    minor_ticks_left=np.linspace(0,0.8,3)
    ax3.fill_between(tmenos25, y1=vmenos25, y2=vmas25,color=color25,alpha=alpha25,label='±25\%')
    ax3.plot(tmenos25,vmenos25, color=color25,linestyle=style25,alpha=alpha25)
    ax3.plot(tmas25,vmas25, color=color25,linestyle=style25,alpha=alpha25)
    
    ax3.fill_between(tmenos5, y1=vmenos5, y2=vmas5,color=color5,alpha=alpha5,label='±5\%')
    ax3.plot(tmenos5,vmenos5, color=color5,linestyle=style5,alpha=alpha5)
    ax3.plot(tmas5,vmas5, color=color5,linestyle=style5,alpha=alpha5)
    
    ax3.plot(tnormal,vnormal,color=colornormal,linestyle=stylenormal,alpha=alphanormal)
    
    
    ax3.grid(which='minor')
    ax3.set_ylabel(r'Volume [L]',size=si)
    ax3.set_xlabel(r'Time [s]',size=si)
    ax3.set_xlim(limxmin,limx)
    ax3.set_yticks([0,0.2, 0.4])
    ax3.set_yticklabels(["0",r"$0.2$", r"$0.4$"], color="k", size=si)
    labelx = -0.15  # axes coords
    ax3.yaxis.set_label_coords(labelx, 0.5)
    ax3.set_xticks([0, 1,2,3])
    ax3.set_xticklabels([r"$0$", r"$1$", r"$2$", r"$3$"], color="k", size=si)
    ax3.set_ylim(0,0.52)
    plt.tight_layout()     
    plt.show()
    plt.savefig('SAmu_vcv.pdf')
    
    return







#%%
    



def plotbaselineVCV():
    print('Baseline')
    folder3='/fo0.69_mu10.33s00'

    name='multi'
    color25='black'
    alpha25=0.2
    style25='-'
    
    color5='black'
    alpha5=0.3
    style5='-'
    
    colornormal='black'
    alphanormal=1
    stylenormal='-'
    

    
    tnormal,pnormal, qnormal,vnormal=give_tpqv(folder3,name)
    
    
    fig, axs = plt.subplots(nrows=3, ncols=1)
    ax1=axs[0]
    ax2=axs[1]
    ax3=axs[2]
    fig.set_size_inches(5, 7)
    
    position='upper right'
    limx=max(tnormal)
    limxmin=-0.02
    si=15
    minor_ticks_left=np.linspace(0,6,4)
    
    

    ax1.plot(tnormal,pnormal*10.2,color=colornormal,linestyle=stylenormal,alpha=alphanormal,label='$      f_0$=0.69  &  $\mu$=10.33')
    
    ax1.set_ylabel(r'Aw. pressure [cm H$_2$O]',size=si)
    ax1.set_xlim(limxmin,limx)
    ax1.set_yticks([0, 4,8])
    ax1.set_yticklabels([r"$0$",r"$4$",r"$8$"], color="k", size=si)
    ax1.tick_params(bottom=False, labelbottom=False)
    ax1.set_ylim(0-0.05,8+0.5)
    labelx = -0.15  # axes coords
    ax1.yaxis.set_label_coords(labelx, 0.5)
    ax1.legend(bbox_to_anchor=(0,1.1,1,0.2),loc='upper left',mode='expand',ncol=6,shadow=True,prop=dict(size=si*0.9)) #bbox_to_anchor=(x0,y0, width, height)
    
    
    
    minor_ticks_left=np.linspace(-80,80,3)
    ax2.plot(tnormal,qnormal,color=colornormal,linestyle=stylenormal,alpha=alphanormal)
    
    
    
    #ax2.set_ylim(-100,100)
    ax2.set_ylabel(r'Flow [L/min]',size=si)
    ax2.axhline(0,color='black',linestyle="-.",lw=0.5)
    ax2.set_xlim(limxmin,limx)
    labelx = -0.15  # axes coords
    ax2.yaxis.set_label_coords(labelx, 0.5)
    ax2.set_yticks([-80, 0, 40])
    ax2.set_yticklabels([r"$-80$", r"$0$", r"$40$"], color="k", size=si)
    ax2.set_ylim(-85,50)
    ax2.tick_params(bottom=False, labelbottom=False)
    
    
    
    minor_ticks_left=np.linspace(0,0.8,3)
    ax3.plot(tnormal,vnormal,color=colornormal,linestyle=stylenormal,alpha=alphanormal)
    
    
    ax3.grid(which='minor')
    ax3.set_ylabel(r'Volume [L]',size=si)
    ax3.set_xlabel(r'Time [s]',size=si)
    ax3.set_xlim(limxmin,limx)
    ax3.set_yticks([0,0.2, 0.4])
    ax3.set_yticklabels(["0",r"$0.2$", r"$0.4$"], color="k", size=si)
    labelx = -0.15  # axes coords
    ax3.yaxis.set_label_coords(labelx, 0.5)
    ax3.set_xticks([0, 1,2,3])
    ax3.set_xticklabels([r"$0$", r"$1$", r"$2$", r"$3$"], color="k", size=si)
    ax3.set_ylim(0,0.52)
    plt.tight_layout()     
    plt.show()
    plt.savefig('Baseline_vcv.pdf')
    
    
    return