% Modal superposition: time domain: piecewise exact integration

Dynamic_analysis;
L=L_span;

P = [ -47600 -76200 -76200; % P1 P2 P3
0 3.70 1.35]; % 0 l2 l3

% P = [ -54000 -103000 -103000; % P1 P2 P3
% 0 3.80 1.385]; % 0 l2 l3

% P = [ -70000 -115000 -115000; % P1 P2 P3
% 0 3.85 1.85]; % 0 l2 l3

DTBB =10; % [m] Distance train/truck to bridge before (front axle)
DTBA =30; % [m] Distance train/truck to bridge after (rear axle)
V = 40*1000/3600; % km/h --> [m/s] Velocity
LT = sum(P(2,:)); % [m] Length of train/truck

seldof=[
[21;22;23;24;25;26;27;28;29;30;31;32;41;42;43;44;45;46;47;48;49;50;51;52]+0.04;
[21;22;23;24;25;26;27;28;29;30;31;32;41;42;43;44;45;46;47;48;49;50;51;52]+0.01;
[21;22;23;24;25;26;27;28;29;30;31;32;41;42;43;44;45;46;47;48;49;50;51;52]+0.05;
[21;22;23;24;25;26;27;28;29;30;31;32;41;42;43;44;45;46;47;48;49;50;51;52]+0.02];

dt=0.001; % Time step/resolution
PLoad = trainload(P,L,DTBB,DTBA,V,dt,seldof,Nodes); % [seldof x N samples]
% Sampling parameters: time domain
T = (DTBB + L + DTBA + LT)/V; % Time window [s]
N=fix(T/dt); % Number of samples 10739
t=[0:N-1]*dt; % Time axis (samples)
t1= DTBB/V; % [s] truck/train enter (firsr axle)
t2 = (DTBB + L + LT)/V; % [s] truck/train leave (last axle)

% Eigenvalue analysis
nMode = 6; % Number of modes to take into account
[phi,omega]=eigfem(K,M,nMode); % Calculate eigenmodes and eigenfrequencies
xi=0.07; % Constant modal damping ratio

% Excitation
pm = zeros(nMode,N);
for itime = 1:N
pm(:,itime) = phi.'*nodalvalues(DOF,seldof,PLoad(:,itime));
end

% Modal excitation
x=msupt(omega,xi,t,pm, 'zoh' );
u=phi*x; % Nodal response (nDOF * N)
% Figures

figure;
plot(t,x);
title('Modal response (piecewise linear exact integration)');
xlabel('Time [s]');
xlim([0 T])
ylabel('Displacement [m kg^{0.5}]');
legend([repmat('Mode ',nMode,1) num2str([1:nMode].')]);
y_xrange = [min(min(x))*1.1 max(max(x))*1.1];
ylim(y_xrange)
t1 = t1(1)*[1;1];
t2 = t2(1)*[1;1];
hold on, plot([t1 t2],y_xrange,'k:','LineWidth',1); hold off; grid on;

figure;
c=selectdof(DOF,[2.03; 6.03; 7.03; 108.03; 11.03]);
plot(t,c*u);
title('Nodal response (piecewise linear exact integration)');
xlabel('Time [s]');
xlim([0 T]); line([0 T*1.1],[0 0],'color','k')
ylabel('Displacement [m]');
legend('2.03','6.03','7.03','108.03','11.03','Location','SouthEast');
y_xrange = [min(min(c*u))*1.1 max(max(c*u))*1.1]; ylim(y_xrange)
hold on, plot([t1 t2],y_xrange,'k:','LineWidth',1); hold off;  grid on;

% % Movie
% figure;
% animdisp(Nodes,Elements,Types,DOF,u);

% Display
disp('Maximum modal response');
disp(max(abs(x),[],2));
disp('Maximum nodal response 2.03; 6.03; 7.03; 108.03; 11.03');
disp(max(abs(c*u),[],2));
