clear;close all
%% set up parameters
addpath("../")
addpath("../baselines");
addpath("../utils")
N1 = 20;
N2 = 25;
N = N1*N2;
p1 = N1*(N1-1)/2;
p2 = N2*(N2-1)/2;
upper = 2; % range of edge weights
lower = 0.1;
model = 'er'; % 'pa', 'ws', 'grid'
filter = 'gmrf';
pd_type = 'strong'; %'tensor'; %
nreplicate = 10;
Ms = [10,100,1000,10000,100000]; % number of graph signals

%% load
filename = join([pd_type,"_prod_",model,"_N1=",num2str(N1),"_N2=",num2str(N2),"_weight=[",num2str(lower),",",num2str(upper),"].mat"],"");
load(filename)

%% set up the baselines
baselines.pst = 0;
baselines.rpgl = 0;
baselines.ff = 0;
baselines.kglasso = 0;
baselines.teralasso = 0;
baselines.mwgl = 0;
baselines.ksgl = 1;
for k = 1:length(Ms)

M = Ms(k);

graphs1_pst = zeros(p1,nreplicate);
graphs1_rpgl = zeros(p1,12,12,nreplicate);
graphs1_ff = zeros(N1^2,nreplicate);
graphs1_kglasso = zeros(N1^2,6,nreplicate);
graphs1_teralasso = zeros(N1^2,12,12,nreplicate);
graphs1_mwgl = zeros(p1,12,nreplicate);
graphs1_ksgl = zeros(p1,8,nreplicate);

graphs2_pst = zeros(p2,nreplicate);
graphs2_rpgl = zeros(p2,12,12,nreplicate);
graphs2_ff = zeros(N2^2,nreplicate);
graphs2_kglasso = zeros(N2^2,6,nreplicate);
graphs2_teralasso = zeros(N2^2,12,12,nreplicate);
graphs2_mwgl = zeros(p2,12,nreplicate);
graphs2_ksgl = zeros(p2,8,nreplicate);

for ii = 1:nreplicate
% parfor (ii = 1:nreplicate,10)

%% generate or load graphs
L_0 = data{ii,2};
Lp1_0 = data{ii,4};
Lp2_0 = data{ii,6};
[X,X_noisy] = generate_signals(L_0,filter,M);
X_M = X(:,1:M);

if baselines.pst == 1
    tic;
    
    param.N1 = N1;
    param.N2 = N2;
    param.template = 'noisy';
    param.pd_type = pd_type;
    param.gso = 'laplacian';
    param.cnt = 1000;
    param.max_iter = 10000;
    param.max_err = 0.05;
    param.delta_err = 0.05;
    param.thr = 0;
    [A,A1,A2] = pst(X_M,param);
    L1 = diag(sum(A1,1))-A1;
    L2 = diag(sum(A2,1))-A2;
    graphs1_pst(:,ii) = -L1(tril(true(N1),-1));
    graphs2_pst(:,ii) = -L2(tril(true(N2),-1));
    
    toc;
end


%% main rpgl loop
if baselines.rpgl == 1
    tic;
    
    beta1 = 0.1.^(0:0.2:2);
    beta2 = 0.1.^(0:0.2:2);
    beta1 = [beta1,0];
    beta2 = [beta2,0];
    len_beta1 = length(beta1);
    len_beta2 = length(beta2);
    
    for i = 1:12
        for j = 1:12
            param = struct();
            param.N1 = N1;
            param.N2 = N2;
            param.solver = 'idqp';
            param.beta1 = beta1(i);
            param.beta2 = beta2(j);
            param.rho = 0.001;
            param.tol = 1e-6;
            param.max_iter = 20000;
            param.thr = 0;
            [L,L1,L2] = rpgl(X_M,param);
            graphs1_rpgl(:,i,j,ii) = -L1(tril(true(N1),-1));
            graphs2_rpgl(:,i,j,ii) = -L2(tril(true(N2),-1));
        end
    end
    
    toc;
end

%% main ff loop
if baselines.ff == 1
    tic;

    S = X_M*X_M'/M;

    niter = 100;
    tol = 1e-5;
    [A, B] = FF(S,N1,N2,niter,tol);
    L1 = inv(A);
    L2 = inv(B);
    graphs1_ff(:,ii) = L1(:);
    graphs2_ff(:,ii) = L2(:);

    toc;
end

%% main kglasso loop
if baselines.kglasso == 1
    tic;

    S = X_M*X_M'/M;
    lambda = [1,0.1,0.01];
    lambda = [lambda,0];
    len_lambda = length(lambda);
    tol = 1e-5;
    niter = 1000;
    for i = 1:4
        [L1,L2] = KGL_iterative(S,N1,N2,M,lambda(i),lambda(i),niter,tol);
        graphs1_kglasso(:,i,ii) = L1(:);
        graphs2_kglasso(:,i,ii) = L2(:);
    end
    toc;
end


%% main teralasso loop
if baselines.teralasso == 1
    tic;
    
    X_M_rs = reshape(X_M,N2,N1,[]);
    T = reshape(X_M_rs,N2,[])*reshape(X_M_rs,N2,[])'/N1/M;
    X_M_rs = permute(X_M_rs,[2,1,3]);
    S = reshape(X_M_rs,N1,[])*reshape(X_M_rs,N1,[])'/N2/M;
    lambda = 0.1.^[2:0.2:4];
    lambda = [lambda,0];
    len_lambda = length(lambda);
    tol = 1e-7;%1e-6;
    maxiter = 1000;
    for i = 1:12
        for j = 1:12
            [PsiH,~ ] = teralasso({S,T},[N1,N2],'L1',1,tol,[lambda(i),lambda(j)],maxiter);
            L1 = PsiH{1};
            L2 = PsiH{2};
            graphs1_teralasso(:,i,j,ii) = L1(:);
            graphs2_teralasso(:,i,j,ii) = L2(:);
        end
    end
    toc;
end

%% main mwgl loop
if baselines.mwgl == 1
    tic;
    
    alpha = 0.1.^(1:0.2:3);
    alpha = [alpha,0];
    len_alpha = length(alpha);
    
    for i = 1:12
    
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.alpha = alpha(i);
        param.pd_type = 'cartesian';
        param.inv_compute = 'eig';
        param.max_iter = 5000;
        param.step_size = 1e-3;
        param.tol = 1e-6;
        [L,L1,L2] = mwgl(X_M,param);
        graphs1_mwgl(:,i,ii) = -L1(tril(true(N1),-1));
        graphs2_mwgl(:,i,ii) = -L2(tril(true(N2),-1));
        
    end
    
    toc;
end

%% main ksgl loop
if baselines.ksgl == 1
    tic;
    
    alpha = 0.1.^(0:0.5:3);
    alpha = [alpha,0];
    len_alpha = length(alpha);
    
    for i = 1:8
    
        param = struct();
        param.N1 = N1;
        param.N2 = N2;
        param.alpha = alpha(i);
        param.pd_type = pd_type;
        param.inv_compute = 'eig';
        param.optim = "alternate";
        param.max_iter = 20;%000;
        param.step_size = 1e-3;
        param.tol = 1e-5;
        while true
            [L,L1,L2] = ksgl(X_M,param);
            if ~any(isnan(L1(:))) && ~any(isnan(L2(:)))
                break
            end
            param.step_size = param.step_size/2;
        end
        graphs1_ksgl(:,i,ii) = -L1(tril(true(N1),-1));
        graphs2_ksgl(:,i,ii) = -L2(tril(true(N2),-1));
        
    end

    toc;
end

end

%% performance

if baselines.pst == 1
    res1_pst_graphs1(:,:,k) = graphs1_pst;
    res2_pst_graphs2(:,:,k) = graphs2_pst;
end

if baselines.rpgl == 1
    res1_rpgl_graphs1(:,:,:,:,k) = graphs1_rpgl;
    res2_rpgl_graphs2(:,:,:,:,k) = graphs2_rpgl;
end

if baselines.ff == 1
    res1_ff_graphs1(:,:,k) = graphs1_ff;
    res2_ff_graphs2(:,:,k) = graphs2_ff;
end

if baselines.kglasso == 1
    res1_kglasso_graphs1(:,:,:,k) = graphs1_kglasso;
    res2_kglasso_graphs2(:,:,:,k) = graphs2_kglasso;
end

if baselines.teralasso == 1
    res1_teralasso_graphs1(:,:,:,:,k) = graphs1_teralasso;
    res2_teralasso_graphs2(:,:,:,:,k) = graphs2_teralasso;
end

if baselines.mwgl == 1
    res1_mwgl_graphs1(:,:,:,k) = graphs1_mwgl;
    res2_mwgl_graphs2(:,:,:,k) = graphs2_mwgl;
end

if baselines.ksgl == 1
    res1_ksgl_graphs1(:,:,:,k) = graphs1_ksgl;
    res2_ksgl_graphs2(:,:,:,k) = graphs2_ksgl;
end

end
