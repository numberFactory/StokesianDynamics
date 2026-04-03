A1 = dlmread('mob_scalars_wall_MB_2562_eig_thresh.txt');
A2 = dlmread('mob_scalars_wall_MB_2562_eig_thresh_new.txt');

for i = 2:6
   subplot(2,3,i-1)
   plot(A1(:,1),A1(:,i))
   hold all
   plot(A2(:,1),A2(:,i),'--')
end