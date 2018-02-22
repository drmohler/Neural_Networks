%David R Mohler
%EE5410: Neural Nets
%HW1: Exhibition of Data Classification (prob 3)


clear 
close all
data = csvread('TrainingPatterns.csv',1,1) 
test_data = csvread('TestPatterns.csv',1,1) 

%Decision Boundary
syms x y
bound = 2.118*x+3.552*y+4.284;

%reorganize training data
[~,idx] = sort(data(:,4)); % sort just the first column
sorteddata = data(idx,:);   % sort the whole matrix using the sort indices


figure(1)
hold on 
%Positive Training data
scatter3(sorteddata(1:14,1),sorteddata(1:14,2),sorteddata(1:14,3),'b')
%Negative Training data
scatter3(sorteddata(15:end,1),sorteddata(15:end,2),sorteddata(15:end,3),'+','g')

%Test Data
scatter3(test_data(:,1),test_data(:,2),test_data(:,3),'d','m')

%Decision Hypersurface (Plane)
fmesh(bound,'EdgeColor','red')
xlim([-2 3])
ylim([-0.5 1.5])
zlim([0 13])  
xlabel('X')
ylabel('Y')
zlabel('Z')
view(45,15)
grid on
legend('+ Training','- Training','Test Data','Boundary')



