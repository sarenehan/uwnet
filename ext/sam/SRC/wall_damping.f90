module wall_damping
  implicit none

contains
  subroutine damp_y_boundary
    use vars
    use grid
    integer i, j, k
    ! 1 hour damping time-scale
    real, parameter :: damping_time_scale = 1200.0
    real, parameter :: damping_distance = 1000e3

    ! damping coefficent local
    real d, yu, yv, du, dv
    integer num_cells_in_bndy, j_north_u, j_north_v

    num_cells_in_bndy = floor(damping_distance/dy)

    print *, 'wall_damping.f90::damping the northern/southern boundaries'

    ! damp zonal velocity
    do k=1,nzm
       do j=1,num_cells_in_bndy
          do i=1,nx+1
             yu = (j - 0.5) * dy
             du = (yu - damping_distance)**2/damping_distance**2 / damping_time_scale
             ! south boundary
             dudt(i,j,k,na) = dudt(i,j,k,na) - du * u(i,j,k)
             ! north boundary
             j_north_u = ny - (j - 1)
             dudt(i,j_north_u,k,na) = dudt(i,j_north_u,k,na) - du * u(i,j_north_u,k)
          end do
        end do
     end do

     ! damp vertical velocity
     do k=1,nz
        do j=1,num_cells_in_bndy
           do i=1,nx
              yu = (j - 0.5) * dy
              du = (yu - damping_distance)**2/damping_distance**2 / damping_time_scale
              ! south boundary
              dwdt(i,j,k,na) = dwdt(i,j,k,na) - du * w(i,j,k)
              ! north boundary
              j_north_u = ny - (j - 1)
              dwdt(i,j_north_u,k,na) = dwdt(i,j_north_u,k,na) - du * w(i,j_north_u,k)
           end do
        end do
     end do

     ! damp meridional
     do k=1,nzm
        do j=1,num_cells_in_bndy
           do i=1,nx
              yv = (j - 1.0) * dy
              dv = (yv - damping_distance)**2/damping_distance**2 / damping_time_scale

              ! south boundary
              dvdt(i,j,k,na) = dvdt(i,j,k,na) - dv * v(i,j,k)

              ! north boundary
              j_north_v = ny - (j - 2)
              dvdt(i,j_north_v,k,na) = dvdt(i,j_north_v,k,na) - dv * v(i,j_north_v,k)
           end do
        end do
     end do
  end subroutine damp_y_boundary

end module wall_damping
