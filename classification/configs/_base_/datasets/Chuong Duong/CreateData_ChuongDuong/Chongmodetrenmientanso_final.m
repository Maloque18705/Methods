% Lexuanthang.official@gmail.com,Lexuanthang.official@outlook.com
% Le Xuan Thang, 2023
% Tutorial: dynamic analysis: direct method: frequency domain
% Units: m, N

%CuaRaoBridge;
L=L_span;

% Train
P = [ -1000000 -2000000 -160000 -300000 -300000 -300000 -300000 -300000 -300000 -300000 -150000 -150000; % P1 P2 P3
0 5 10 20 20 20 20 20 20 20 20 20]; % 0 l2 l3

DTBB =50; % [m] Distance train/truck to bridge before (front axle)
DTBA =70; % [m] Distance train/truck to bridge after (rear axle)
V = 150*1000/3600; % km/h --> [m/s] Velocity
LT = sum(P(2,:)); % [m] Length of train/truck

seldof=reprow([100,300,500,600],1,8,[1,1,1,1])+0.03;
seldof = seldof(:);

t1= DTBB/V; % [s] truck/train enter (firsr axle)
t2 = (DTBB + L + LT)/V; % [s] truck/train leave (last axle)

% PLoad = awgn(PLoad,0.5,'measured','linear');
% Sampling parameters: time domain
T = 1000; % Time window [s]
dt= 1; % Time step/resolution
N = fix(T/dt); % Number of samples 10739

% Tạo trục thời gian
t=(0:N-1)*dt; % Time axis (samples)

% Gọi hàm trainload với tham số kích thích được điều chỉnh
% Thời gian bắt đầu kích thích (firstime) cứ mỗi 100 giây
timestart = 2;
Pulse = -10000;
% PLoad = trainload(P, L, DTBB, DTBA, V, dt, seldof, Nodes, firstime, Pulse, T);
PLoad = trainload(P,L,DTBB,DTBA, V,dt,seldof,Nodes,timestart,T); % [seldof x N samples]
figure(3)
plot(PLoad(1,:))


% Sampling parameters: frequency domain
F=1/dt; % Sampling frequency [Hz]
df=1/T; % Frequency resolution
f=(0:fix(N/2)-1)*df; % Positive frequencies corresponding to FFT [Hz]
Omega=2*pi*f; % [rad/s] excitation frequency content

% Eigenvalue analysis
nMode = 6; % Number of modes to take into account
[phi,omega]=eigfem(K,M,nMode); % Calculate eigenmodes and eigenfrequencies
xi=0.07; % Constant modal damping ratio
% Excitation: transfer PLoad vector to nodalforce vector
% (seldof --> all DOF)
Pnodal = zeros(size(DOF,1),N);
for itime = 1:N
    Pnodal(:,itime) = nodalvalues(DOF,seldof,PLoad(:,itime));
end

% Modal excitation
q = zeros(size(DOF,1),N);
q(:,(t>=0.5)&(t<=0.6)) = -1000000+Pnodal(:,(t>=0.5)&(t<=0.6));

Pnodal = q + Pnodal;
figure(4)
plot(Pnodal(5,:))
Pm_ = phi.'*Pnodal; % [DOF,nMode] x [DOF, N samples]

% Transfer nodal force vector time history to frequency domain
Q = zeros(nMode,fix(N/2)); % keep positive frequency ONLY
for inMode = 1:nMode
temp = fft(Pm_(inMode,:));
Q(inMode,:) = temp(1:fix(N/2));
end
Pm = Q;
% Modal analysis: calc. the modal transfer functions and the modal disp.
[X,H]=msupf(omega,xi,Omega,Pm); % Modal response, positive freq (nMode * N/2)
% F-dom -> t-dom [inverse Fourier transform]
X = [X, zeros(nMode,1), conj(X(:,end:-1:2))];
x = ifft(X,[],2); % Modal response (nMode * N)
% Modal displacements -> nodal displacements
u=phi*x; % Nodal response (nDOF * N)
% Figures
figure;
subplot(2,2,1);
plot(t,[PLoad([3,6],:);sum(PLoad)],'.-');
xlim([0 T])
ylim('auto');
title('Excitation time history');
xlabel('Time [s]');
ylabel('Force [N]');

grid on

subplot(2,2,2);
plot(f,abs(Q([1,2,3,4,5,6],:))/F,'.-');
title('Excitation frequency content');
xlabel('Frequency [Hz]');
ylabel('Force [N/Hz]');
xlim([0 3])
legend([repmat('Mode ',6,1) num2str([1,2,3,4,5,6].')]);
grid on

subplot(2,2,3);
plot(f,abs(H([1,2,3,4,5,6],:)),'.-');
title('Modal transfer function');
xlabel('Frequency [Hz]');
ylabel('Displacement [m/N]');
xlim([0 10])
legend([repmat('Mode ',6,1) num2str([1,2,3,4,5,6].')]);grid on

subplot(2,2,4);
plot(f,abs(X(:,1:fix(N/2)))/F,'.-');
xlim([0 3])
title('Modal response');
xlabel('Frequency [Hz]');
ylabel('Displacement [m kg^{0.5}/Hz]');grid on
legend([repmat('Mode ',6,1) num2str([1,2,3,4,5,6].')]);



figure;
plot(t,x);
title('Modal response (calculation in f-dom)');
xlabel('Time [s]');
xlim([0 T])
ylabel('Displacement [m kg^{0.5}]');
legend([repmat('Mode ',nMode,1) num2str((1:nMode).')]);

figure;
c=selectdof(DOF,[104.03]);
plot(t,c*u);
title('Nodal response (computed in the frequency domain)');
xlabel('Time [s]');
xlim([0 T])
ylabel('Displacement [m]');grid on
legend('104.03','6.03','7.03','108.03','11.03');

% Movie
% figure;
% animdisp(Nodes,Elements,Types,DOF,u);
% Display
disp('Maximum modal response');
disp(max(abs(x),[],2));
disp('Maximum nodal response 2.03; 6.03; 7.03; 108.03; 11.03');
disp(max(abs(c*u),[],2));
% tinh =f(max(abs(c*u),[],2)
% chuahong=[0.0010
%     0.0033
%     0.0028]
% error = norm(tinh-chuahong.)./chuahong
