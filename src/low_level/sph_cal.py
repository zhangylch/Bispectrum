import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from functools import partial
import jax.random as jrm
import scipy

class SPH_CAL():
    '''
    This module perform the calculation of spherical harmonic expansion based on the derivation in this work (https://arxiv.org/abs/1410.1748).
    max_l: int32/int64
         represents the maximal angular quantum numebr for the evaluation of spherical harmonic. Example: 2

    Dtype: jnp.float32/jnp.float64
         represents the datatype in this module. Example: jnp.float32
    '''
    def __init__(self,max_l=3,Dtype=jnp.dtype("float32")):
        if max_l<1: raise ValueError("The angular momentum must be greater than or equal to 1. Or the angular momentum is lack of angular information, the calculation of the sph is meanless.")
        self.max_l=max_l+1
        self.Dtype=Dtype
        self.pt=np.empty((self.max_l,self.max_l),dtype=np.int64)
        self.yr=np.empty((self.max_l,self.max_l),dtype=np.int64)
        self.yr_rev=np.empty((self.max_l,self.max_l),dtype=np.int64)
        self.power=np.power(-1,np.arange(self.max_l))
        num_lm=int((self.max_l+1)*self.max_l/2)
        coeff_a=np.empty(num_lm,dtype=Dtype)
        coeff_b=np.empty(num_lm,dtype=Dtype)
        tmp=jnp.arange(self.max_l)
        self.prefactor1=-jnp.sqrt(1.0+0.5/tmp).astype(Dtype)
        self.prefactor2=jnp.sqrt(2.0*tmp+3).astype(Dtype)
        tmp=np.arange(self.max_l)
        ls=tmp*tmp
        for l in range(self.max_l):
            self.pt[l,0:l+1]=tmp[0:l+1]+int(l*(l+1)/2)
            # here the self.yr and self.yr_rev have overlap in m=0.
            self.yr[l,0:l+1]=ls[l]+l+tmp[0:l+1]
            self.yr_rev[l,0:l+1]=ls[l]+l-tmp[0:l+1]
            if l>0.5:
                coeff_a[self.pt[l,0:l]]=np.sqrt((4.0*ls[l]-1)/(ls[l]-ls[0:l]))
                coeff_b[self.pt[l,0:l]]=-np.sqrt((ls[l-1]-ls[0:l])/(4.0*ls[l-1]-1.0))

        self.sqrt2_rev=np.sqrt(1/2.0).astype(Dtype)
        self.sqrt2pi_rev=np.sqrt(0.5/np.pi).astype(Dtype)
        self.hc_factor1=np.sqrt(15.0/4.0/np.pi).astype(Dtype)
        self.hc_factor2=np.sqrt(5.0/16.0/np.pi).astype(Dtype)
        self.hc_factor3=np.sqrt(15.0/16.0/np.pi).astype(Dtype)
        self.coeff_a=jnp.array(coeff_a)
        self.coeff_b=jnp.array(coeff_b)


    @partial(jit,static_argnums=0)
    def __call__(self,cart):
        '''
        cart: jnp.float32/jnp.float64.
            represents the cartesian coordinates of systems with its shape [3,...]. 
        '''
        distances=jnp.linalg.norm(cart,axis=0)  # to convert to the dimension (n,batchsize)
        d_sq=distances*distances
        sph_shape=(self.max_l*self.max_l,)+cart.shape[1:]
        sph=jnp.empty(sph_shape,dtype=cart.dtype)
        sph_shape=(2,)+sph_shape
        sph_complex=jnp.zeros(sph_shape,dtype=cart.dtype)
        sph=sph.at[0].set(self.sqrt2pi_rev*self.sqrt2_rev)
        sph_complex=sph_complex.at[0,0].set(sph[0])
        sph=sph.at[1].set(self.prefactor1[1]*self.sqrt2pi_rev*cart[1])
        sph=sph.at[2].set(self.prefactor2[0]*self.sqrt2_rev*self.sqrt2pi_rev*cart[2])
        sph=sph.at[3].set(self.prefactor1[1]*self.sqrt2pi_rev*cart[0])
        #==================complex sph from real sph=============================
        l=1
        sph_complex=sph_complex.at[0,self.yr[l,0:l+1]].set(jnp.einsum("i,i... -> i...",self.power[0:l+1],sph[self.yr_rev[l,0:l+1]]))
        sph_complex=sph_complex.at[1,self.yr[l,1:l+1]].set(jnp.einsum("i,i... -> i...",self.power[0:l],sph[self.yr[l,1:l+1]]))
        sph_complex=sph_complex.at[0,self.yr_rev[l,1:l+1]].set(jnp.einsum("i,i... -> i...",self.power[0:l],sph[self.yr_rev[l,1:l+1]]))
        sph_complex=sph_complex.at[1,self.yr_rev[l,1:l+1]].set(jnp.einsum("i,i... -> i...",self.power[0:l],sph[self.yr[l,1:l+1]]))
        if self.max_l>2.5:
            sph=sph.at[4].set(self.hc_factor1*cart[0]*cart[1])
            sph=sph.at[5].set(-self.hc_factor1*cart[1]*cart[2])
            sph=sph.at[6].set(self.hc_factor2*(3.0*cart[2]*cart[2]-d_sq))
            sph=sph.at[7].set(-self.hc_factor1*cart[0]*cart[2])
            sph=sph.at[8].set(self.hc_factor3*(cart[0]*cart[0]-cart[1]*cart[1]))
            #==================complex sph from real sph=============================
            l=2
            sph_complex=sph_complex.at[0,self.yr[l,0:l+1]].set(jnp.einsum("i,i... -> i...",self.power[0:l+1],sph[self.yr_rev[l,0:l+1]]))
            sph_complex=sph_complex.at[1,self.yr[l,1:l+1]].set(jnp.einsum("i,i... -> i...",self.power[0:l],sph[self.yr[l,1:l+1]]))
            sph_complex=sph_complex.at[0,self.yr_rev[l,1:l+1]].set(jnp.einsum("i,i... -> i...",self.power[0:l],sph[self.yr_rev[l,1:l+1]]))
            sph_complex=sph_complex.at[1,self.yr_rev[l,1:l+1]].set(jnp.einsum("i,i... -> i...",self.power[0:l],sph[self.yr[l,1:l+1]]))
            for l in range(3,self.max_l):
                sph=sph.at[self.yr[l,0:l-1]].set(jnp.einsum("i,i...->i...",self.coeff_a[self.pt[l,0:l-1]],(cart[2]*sph[self.yr[l-1,0:l-1]]+jnp.einsum("i,...,i... ->i...",self.coeff_b[self.pt[l,0:l-1]],d_sq,sph[self.yr[l-2,0:l-1]]))))
                sph=sph.at[self.yr_rev[l,1:l-1]].set(jnp.einsum("i,i... ->i...",self.coeff_a[self.pt[l,1:l-1]],(cart[2]*sph[self.yr_rev[l-1,1:l-1]]+jnp.einsum("i,...,i... ->i...",self.coeff_b[self.pt[l,1:l-1]],d_sq,sph[self.yr_rev[l-2,1:l-1]]))))
                sph=sph.at[self.yr[l,l-1]].set(self.prefactor2[l-1]*cart[2]*sph[self.yr[l-1,l-1]])
                sph=sph.at[self.yr_rev[l,l-1]].set(self.prefactor2[l-1]*cart[2]*sph[self.yr_rev[l-1,l-1]])
                sph=sph.at[self.yr[l,l]].set(self.prefactor1[l]*(cart[0]*sph[self.yr[l-1,l-1]]-cart[1]*sph[self.yr_rev[l-1,l-1]]))
                sph=sph.at[self.yr_rev[l,l]].set(self.prefactor1[l]*(cart[0]*sph[self.yr_rev[l-1,l-1]]+cart[1]*sph[self.yr[l-1,l-1]]))
                sph_complex=sph_complex.at[0,self.yr[l,0:l+1]].set(jnp.einsum("i,i... -> i...",self.power[0:l+1],sph[self.yr_rev[l,0:l+1]]))
                sph_complex=sph_complex.at[1,self.yr[l,1:l+1]].set(jnp.einsum("i,i... -> i...",self.power[0:l],sph[self.yr[l,1:l+1]]))
                sph_complex=sph_complex.at[0,self.yr_rev[l,1:l+1]].set(jnp.einsum("i,i... -> i...",self.power[0:l],sph[self.yr_rev[l,1:l+1]]))
                sph_complex=sph_complex.at[1,self.yr_rev[l,1:l+1]].set(jnp.einsum("i,i... -> i...",self.power[0:l],sph[self.yr[l,1:l+1]]))
        return jax.lax.complex(sph_complex[0],sph_complex[1])

# here is an example to use the sph calculation
import wigners
max_l=4
key=jrm.PRNGKey(0)
init_key=jrm.split(key)
cart=jrm.uniform(key,(3,10))
rotate=jnp.zeros((3,3))
ceta =np.pi*60.0/180.0
rotate=rotate.at[0,0].set(jnp.cos(ceta))
rotate=rotate.at[1,1].set(jnp.cos(ceta))
rotate=rotate.at[0,1].set(jnp.sin(ceta))
rotate=rotate.at[1,0].set(-jnp.sin(ceta))
rotate=rotate.at[2,2].set(1.0)
cart1=jnp.einsum("ij,jk->ik",rotate,cart)
sph=SPH_CAL(max_l=max_l)

def contraction(cart,j1,j2,j3):
    sph_complex=sph(cart)
    sum_sph=jnp.sum(sph_complex,axis=1)
    feature=0
    num1=j1*j1
    num2=j2*j2
    num3=j3*j3
    for m1 in range(-j1,j1+1):
        for m2 in range(-j2,j2+1):
            for m3 in range(-j3,j3+1):
                if m1+m2+m3==0: feature+=sum_sph[num1+j1+m1]*sum_sph[num2+j2+m2]*sum_sph[num3+j3+m3]*wigners.wigner_3j(j1,j2,j3,m1,m2,m3)
    return feature

print(contraction(cart,1,1,2))
print(contraction(cart1,1,1,2))

#print(jax.make_jaxpr(sph.compute_sph)(cart))
'''dd
jax.lax.stop_gradient(cart)
starttime = timeit.default_timer()
print("The start time is :",starttime)
tmp=sph(cart)       
print("The time difference is :", timeit.default_timer() - starttime)
print(tmp)
print(cart.reshape(-1))
forward=jax.jit(jax.vmap(test_forward,in_axes=(1),out_axes=(1)))
forward(cart)
starttime = timeit.default_timer()
print("The start time is :",starttime)
tmp=forward(cart)
print("The time difference is :", timeit.default_timer() - starttime)
print(tmp.shape)

#jac=jax.jit(jax.vmap(jax.jacfwd(test_forward),in_axes=(1),out_axes=(1)))
#grad=jac(cart)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#grad=jac(cart)
#print("The time difference is :", timeit.default_timer() - starttime)

#hess=jax.jit(jax.vmap(jax.hessian(sph.compute_sph),in_axes=(1),out_axes=(1)))
#tmp=hess(cart)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#tmp=hess(cart)
#print("The time difference is :", timeit.default_timer() - starttime)
#
## calculate hessian by jac(jac)
#hess=jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(sph.compute_sph)),in_axes=(1),out_axes=(1)))
#tmp=hess(cart)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#tmp=hess(cart)
#print("The time difference is :", timeit.default_timer() - starttime)
#print(tmp.shape)
'''
