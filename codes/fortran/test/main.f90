program main
    use constant
    use initmod
    implicit none
    integer(kind=intype),parameter :: numatom=5,maxneigh=20
    real(kind=typenum),parameter :: in_rc=5.0,in_dier=5.0
    integer(kind=intype) i,j,scutnum
    integer(kind=intype) :: atomindex(2,maxneigh)
    real(kind=typenum) :: shifts(3,maxneigh)
    real(kind=typenum) ::  cart(3,numatom),cell(3,3)
    cell=0d0
    cell(1,1)=25.0
    cell(2,2)=25.0
    cell(3,3)=25.0
    call random_number(cart)
    cart=cart*10
    do i=1,numatom-1
      do j=i+1,numatom
         write(*,*) dsqrt(dot_product(cart(:,i)-cart(:,j),cart(:,i)-cart(:,j)))
      end do
    end do
    call init_neigh(in_rc,in_dier,cell)
    call get_neigh(cart,atomindex,shifts,maxneigh,numatom,scutnum)
    write(*,*) atomindex,shifts
    call deallocate_all()
end 
