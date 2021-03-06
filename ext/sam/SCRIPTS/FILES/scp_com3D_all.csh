#! /bin/csh -f

set filename = /gpfs/scratch1/marat/SAM6.7_SR/OUT_3D/GATE_IDEAL_S_2048x2048x256_100m_2s_2048_00000
set dirout = ux454536@dslogin.sdsc.edu:/gpfs-wan/scratch/ux454536/SAM6.7_SR/OUT_3D
set N1 = 7200
set N2 = 43200
set NN = 450
set NC  = 8  # niumber of scp channels
@ N2 = $N2 - $NN * ($NC - 1) 
echo $N2

while ($N1 < $N2)

  set CH = 0

  while ($CH < $NC)

   @ N1 = $N1 + $NN
   @ CH = $CH + 1
   set M = ""
   if($N1 < 10) then
    set M = "0000"
   else if($N1 < 100) then
    set M = "000"
   else if($N1 < 1000) then
    set M = "00"
   else if($N1 < 10000) then
    set M = "0"
   endif


   set f = $filename$M$N1.com3D
   if($CH == $NC) then
     scp $f $dirout 
   else
     scp $f $dirout &
   endif

  end

  set COUNT=1
  while ($COUNT != 0)
   sleep 10
   set COUNT = `ps -ef | grep $filename |  grep scp | wc -l`
  end


end

