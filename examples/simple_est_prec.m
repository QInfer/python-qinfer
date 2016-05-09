% simple_est_prec.m: Uses QInfer to estimate simple precession
%     frequencies.

% We must set this environment variable to work around an internal MATLAB
% bug.
setenv MKL_NUM_THREADS 1

true_omega = 70.3;
omega_min = 0;
omega_max = 99.1;
n_shots = 400;

ts = pi * (1:1:100) / (2 * omega_max);

signal = sin(true_omega * ts / 2) .^ 2;
counts = binornd(n_shots, signal);

data = py.numpy.column_stack({counts ts n_shots * ones(1, size(ts, 2))});
est = py.qinfer.simple_est_prec(data, pyargs('freq_min', omega_min, 'freq_max', omega_max));
mu = est{1};
sigma = est{2};

relative_error = abs(mu - true_omega) / true_omega
