import sys
import numpy as np
import jax
import flax 
import jax.numpy as jnp
import jax.random as jrm
# here is an example to use the sph calculation
sys.path.append("../")
import Bispectrum
import fortran.getneigh as getneigh

cutoff=5.0
nwave=2
max_l=5
numatom=8
maxneigh=80
key=jrm.PRNGKey(0)
emb_nl=[4,4]
dtype=jnp.dtype("float32")
cart=(np.random.rand(3,numatom)*3).astype(dtype)
species=jnp.array([12,1,1,1,3,5,3,1]).reshape(-1,1)
cell=jnp.zeros((3,3),dtype=dtype)
cell=cell.at[0,0].set(100.0)
cell=cell.at[1,1].set(100.0)
cell=cell.at[2,2].set(100.0)
atomindex=np.ones((2,maxneigh),dtype=np.int32)
shifts=np.ones((3,maxneigh),dtype=dtype)
in_dier=cutoff/2.0
getneigh.init_neigh(cutoff,in_dier,cell)

cart,atomindex,shifts,scutnum=getneigh.get_neigh(cart,maxneigh)
getneigh.deallocate_all()
cart=jnp.array(cart)
#jax.config.update("jax_debug_nans", True)

J_seq=[[2,3,3],[2,3,5]]
model=Bispectrum.Bispectrum(emb_nl,J_seq,nwave=nwave,max_l=max_l,cutoff=cutoff,Dtype=dtype)
params=model.init(key,cart,atomindex,shifts,species)
model=model.apply
energy=model(params,cart,atomindex,shifts,species)
print(energy)
rotate=jnp.zeros((3,3),dtype=dtype)
ceta=np.pi/4
rotate=rotate.at[2,2].set(1.0)
rotate=rotate.at[1,1].set(jnp.cos(ceta))
rotate=rotate.at[0,0].set(jnp.cos(ceta))
rotate=rotate.at[0,1].set(jnp.sin(ceta))
rotate=rotate.at[1,0].set(-jnp.sin(ceta))
cart=jnp.einsum("ij,jk->ik",rotate,cart)
energy1=model(params,cart,atomindex,shifts,species)
print(energy1)

