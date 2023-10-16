import sys
import numpy as np
import jax
import flax 
import jax.numpy as jnp
import jax.random as jrm
from jax.numpy import dtype,array
from typing import Sequence
import sph_cal
import MLP
from wigners import wigner_3j
import flax.linen as nn
import wigners

class Bispectrum(nn.Module):

    '''
    This Bispectrum module is designed for the calculation of four-body descriptors for both periodic and molecular systems.

    emb_nl: list
        defines the nn structure (only the hidden layers) of embedding neural network. Examples: [16,16]

    nwave: int32/int64
         represents the number of gaussian radial functions. Example: 8

    max_l: int32/int64
         represents the maximal angular quantum numebr for the evaluation of spherical harmonic. Example: 2

    cutoff: float32/float64
         represents the cutoff radius for evaluating the local descriptors. Example: 4.0

    Dtype: jnp.float32/jnp.float64
         represents the datatype in this module. Example: jnp.float32
         
    J_eq: jnp.int32/jnp.int64
         represents the angular momentum used for calculating the Bispectrum. Note here, the maximal of the three value should be less than the sum of the rest two j.
    '''  

    emb_nl: Sequence[int]
    J_seq: Sequence[int]
    nwave: int=8
    max_l: int=2
    cutoff: float=5.0
    Dtype: dtype=dtype(jnp.float32)
    

    def setup(self):
        # Sequence of angular moment
        self.nseq=len(self.J_seq)
        self.r_max_l=self.max_l+1

        # define the class for the calculation of spherical harmonic expansion
        self.sph_cal=sph_cal.SPH_CAL(max_l=self.max_l,Dtype=self.Dtype)
        # the first time is slow for the compile of the jit
        self.sph_cal(jnp.ones(3))
        
        # define the embedded layer used to convert the atomin number of a coefficients
        self.emb_nn=MLP.MLP(self.emb_nl,self.nwave*3)
        self.emb_params=self.param("emb_params",self.emb_nn.init,(jnp.ones(1)))

       

    def __call__(self,cart,atomindex,shifts,species):
        '''
        cart: jnp.float32/jnp.float64.
            represents the cartesian coordinates of systems with dimension 3*Natom. Natom is the number of atoms in the system.

        atomindx: jnp.int64/inp.int32
            stores the index of centeral atoms and theirs corresponding neighbor atoms with the dimension 2*Neigh. Neigh is the number of total neighbor atoms in the system.

        shifts: jnp.float32/jnp.float64
            stores the offset corresponding to each neighbor atoms with the dimension Neigh*3.

        species: jnp.float32/jnp.float64
            represents the atomic number of each center atom with the dimension Natom.
        '''
        coor=cart[:,atomindex[1]]-cart[:,atomindex[0]]+shifts
        distances=jnp.linalg.norm(coor,axis=0)
        emb_coeff=self.emb_nn.apply(self.emb_params,species)
        expand_coeff=emb_coeff[atomindex[1]]
        coefficients=expand_coeff[:,:self.nwave]
        alpha=expand_coeff[:,self.nwave:2*self.nwave]
        center=expand_coeff[:,2*self.nwave:]
        radial=self.gaussian(distances,alpha,center)
        radial_cutoff=self.cutoff_func(distances)
        sph=self.sph_cal(coor/distances)
        equi_feature = jnp.einsum("i,ij,ij,ki -> ikj",radial_cutoff,radial,coefficients,sph)
        sum_sph=jnp.zeros((cart.shape[1],sph.shape[0],self.nwave),dtype=sph.dtype)
        sum_sph=sum_sph.at[atomindex[0]].add(equi_feature)
        sum_sph=sum_sph.transpose(1,0,2)
        density=jnp.zeros((self.nseq,cart.shape[1],self.nwave),dtype=sph.dtype)
        for iJ in range(self.nseq):
            J_seq=self.J_seq[iJ]
            tmp=jnp.array(J_seq)
            num=(tmp+1)*tmp
            for m1 in range(-J_seq[0],J_seq[0]+1):
                m2_down=max(-J_seq[2]-m1,-J_seq[1])
                m2_up=min(J_seq[2]-m1+1,J_seq[1]+1)
                for m2 in range(m2_down,m2_up):
                    m3=-m1-m2
                    density=density.at[iJ].add(sum_sph[num[0]+m1]*sum_sph[num[1]+m2]*sum_sph[num[2]+m3]*wigners.wigner_3j(J_seq[0],J_seq[1],J_seq[2],m1,m2,m3))
        return jnp.real(density)

    def gaussian(self,distances,alpha,center):
        '''
        gaussian radial functions
        '''
        shift_distances=alpha*(distances[:,None]-center)
        gaussian=jnp.exp(-shift_distances*shift_distances)
        return gaussian
    
    def cutoff_func(self,distances):
        '''
        cutoff function:
        '''
        tmp=(jnp.cos(distances/self.cutoff*jnp.pi)+1.0)/2.0
        return tmp*tmp*tmp  # here to use the a^3 to keep the smooth of hessian functtion

