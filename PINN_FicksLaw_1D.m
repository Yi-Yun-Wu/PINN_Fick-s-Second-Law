clc; clear;

%% 參數設定
D = 0.1;     % 擴散係數
L = 1.0;     % 空間長度
Tmax = 1.0;  % 模擬最長時間

%% 生成訓練資料點
N_interior = 10000;
x_c = L * rand(N_interior,1);
t_c = Tmax * rand(N_interior,1);

N0 = 500;
x0 = L * rand(N0,1);
t0 = zeros(N0,1);
C0 = sin(pi*x0);  % 初始條件

dlX = dlarray(x_c','CB');
dlT = dlarray(t_c','CB');
dlX0 = dlarray(x0','CB');
dlT0 = dlarray(t0','CB');
dlC0 = dlarray(C0','CB');

%% 建立神經網路
numLayers = 8;
numNeurons = 20;
layers = layerGraph();
layers = addLayers(layers, featureInputLayer(2,"Name","input"));

for i = 1:numLayers
    fcName = "fc" + i;
    actName = "tanh" + i;
    layers = addLayers(layers, fullyConnectedLayer(numNeurons,"Name",fcName));
    layers = addLayers(layers, tanhLayer("Name",actName));
    if i == 1
        layers = connectLayers(layers, "input", fcName);
    else
        layers = connectLayers(layers, "tanh" + (i-1), fcName);
    end
    layers = connectLayers(layers, fcName, actName);
end

layers = addLayers(layers, fullyConnectedLayer(1,"Name","output"));
layers = connectLayers(layers, "tanh" + numLayers, "output");
net = dlnetwork(layers);

%% 損失函數
function [loss,gradients] = modelLoss(net,dlX,dlT,dlX0,dlT0,dlC0,D)
XT = [dlX; dlT];
C = forward(net, XT);

dCdt = dlgradient(sum(C,'all'), dlT, 'EnableHigherDerivatives', true);
dCdx = dlgradient(sum(C,'all'), dlX, 'EnableHigherDerivatives', true);
d2Cdx2 = dlgradient(sum(dCdx,'all'), dlX, 'EnableHigherDerivatives', true);

GE = dCdt - D * d2Cdx2;
mseF = mean(GE.^2);

C0_pred = forward(net, [dlX0; dlT0]);
mse0 = mean((C0_pred - dlC0).^2);

loss = mseF + mse0;
gradients = dlgradient(loss, net.Learnables);
end

%% 訓練
solverState = lbfgsState;
maxIter = 3000;
lossFcn = @(net) dlfeval(@modelLoss, net, dlX, dlT, dlX0, dlT0, dlC0, D);
monitor = trainingProgressMonitor("Metrics", "Loss", "XLabel", "Iteration");

iteration = 0;
while iteration < maxIter && ~monitor.Stop
    iteration = iteration + 1;
    [net, solverState] = lbfgsupdate(net, lossFcn, solverState);
    monitor.Progress = 100 * iteration / maxIter;
    recordMetrics(monitor, iteration, Loss=solverState.Loss);
    if solverState.LineSearchStatus == "failed"
        break
    end
end

%% 預測與繪圖
Nx = 100; Nt = 100;
xv = linspace(0, L, Nx);
tv = linspace(0, Tmax, Nt);
[XX, TT] = meshgrid(xv, tv);
Xgrid = reshape(XX,1,[]);
Tgrid = reshape(TT,1,[]);
XT_dl = dlarray([Xgrid; Tgrid],'CB');

C_pred = predict(net, XT_dl);
C_plot = reshape(extractdata(C_pred), Nt, Nx);

figure;
surf(XX, TT, C_plot, 'EdgeColor', 'none');
view(45, 30);  % 設定等角視角
xlabel('x'); ylabel('t'); zlabel('C(x,t)');
title('C(x,t) 等角濃度圖');
colorbar;
shading interp;  % 平滑色階

