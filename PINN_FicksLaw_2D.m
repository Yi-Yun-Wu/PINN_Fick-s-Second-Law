clc; clear;

%% 參數設定
D = 0.1;     % 擴散係數
L = 1.0;     % 空間長度
Tmax = 1.0;  % 模擬最長時間

%% 生成訓練資料點
N_interior = 10000;
x_c = L * rand(N_interior,1);
y_c = L * rand(N_interior,1);
t_c = Tmax * rand(N_interior,1);

N0 = 1000;
x0 = L * rand(N0,1);
y0 = L * rand(N0,1);
t0 = zeros(N0,1);
C0 = sin(pi*x0).*sin(pi*y0);  % 初始條件

dlX = dlarray(x_c','CB');
dlY = dlarray(y_c','CB');
dlT = dlarray(t_c','CB');
dlX0 = dlarray(x0','CB');
dlY0 = dlarray(y0','CB');
dlT0 = dlarray(t0','CB');
dlC0 = dlarray(C0','CB');

%% 建立神經網路
numLayers = 8;
numNeurons = 20;
layers = layerGraph();
layers = addLayers(layers, featureInputLayer(3,"Name","input"));

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
function [loss,gradients] = modelLoss(net,dlX,dlY,dlT,dlX0,dlY0,dlT0,dlC0,D)
XYZT = [dlX; dlY; dlT];
C = forward(net, XYZT);

dCdt = dlgradient(sum(C,'all'), dlT, 'EnableHigherDerivatives', true);
dCdx = dlgradient(sum(C,'all'), dlX, 'EnableHigherDerivatives', true);
dCdy = dlgradient(sum(C,'all'), dlY, 'EnableHigherDerivatives', true);
d2Cdx2 = dlgradient(sum(dCdx,'all'), dlX, 'EnableHigherDerivatives', true);
d2Cdy2 = dlgradient(sum(dCdy,'all'), dlY, 'EnableHigherDerivatives', true);

GE = dCdt - D * (d2Cdx2 + d2Cdy2);
mseF = mean(GE.^2);

C0_pred = forward(net, [dlX0; dlY0; dlT0]);
mse0 = mean((C0_pred - dlC0).^2);

loss = mseF + mse0;
gradients = dlgradient(loss, net.Learnables);
end

%% 訓練
solverState = lbfgsState;
maxIter = 5000;
lossThreshold = 1e-4;
waitCount = 0;
maxWait = 500;
lossFcn = @(net) dlfeval(@modelLoss, net, dlX, dlY, dlT, dlX0, dlY0, dlT0, dlC0, D);
monitor = trainingProgressMonitor("Metrics", "Loss", "XLabel", "Iteration");

iteration = 0;
while iteration < maxIter && ~monitor.Stop
    iteration = iteration + 1;
    [net, solverState] = lbfgsupdate(net, lossFcn, solverState);
    monitor.Progress = 100 * iteration / maxIter;
    recordMetrics(monitor, iteration, Loss=solverState.Loss);

    if solverState.Loss < lossThreshold
        waitCount = waitCount + 1;
        if waitCount >= maxWait
            break;
        end
    else
        waitCount = 0;
    end

    if solverState.LineSearchStatus == "failed"
        break;
    end
end

%% 預測與繪圖與GIF輸出
Nx = 50; Ny = 50;
xv = linspace(0, L, Nx);
yv = linspace(0, L, Ny);
t_samples = linspace(0, Tmax, 100);

filename = 'C_2D_Diffusion.gif';

% 預先決定全域 Z 軸與色階範圍
Zmin = 0; Zmax = 1;

snapshot_times = [0.0, 0.3, 0.6, 1.0];

for k = 1:length(t_samples)
    t_now = t_samples(k);
    [XX, YY] = meshgrid(xv, yv);
    TT = t_now * ones(size(XX));

    Xgrid = reshape(XX,1,[]);
    Ygrid = reshape(YY,1,[]);
    Tgrid = reshape(TT,1,[]);
    XYT_dl = dlarray([Xgrid; Ygrid; Tgrid],'CB');

    C_pred = predict(net, XYT_dl);
    C_plot = reshape(extractdata(C_pred), Ny, Nx);

    fig = figure('visible','off');
    surf(XX, YY, C_plot, 'EdgeColor', 'none');
    view(45, 30);
    xlabel('x'); ylabel('y'); zlabel('C(x,y,t)');
    title(['C(x,y,t) at t = ', num2str(t_now, '%.2f')]);
    zlim([Zmin Zmax]);
    caxis([Zmin Zmax]);
    colormap(jet);
    colorbar;
    shading interp;

    frame = getframe(fig);
    img = frame2im(frame);
    [A,map] = rgb2ind(img,256);
    if k == 1
        imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',0.04);
    else
        imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',0.04);
    end
    close(fig);

    % 輸出指定時刻的圖片
    if ismember(round(t_now, 2), snapshot_times)
        fig2 = figure;
        surf(XX, YY, C_plot, 'EdgeColor', 'none');
        view(45, 30);
        xlabel('x'); ylabel('y'); zlabel('C(x,y,t)');
        title(['C(x,y,t) at t = ', num2str(t_now, '%.2f')]);
        zlim([Zmin Zmax]);
        caxis([Zmin Zmax]);
        colormap(jet);
        colorbar;
        shading interp;
        saveas(fig2, sprintf('C_2D_t%.2f.png', t_now));
        close(fig2);
    end
end
