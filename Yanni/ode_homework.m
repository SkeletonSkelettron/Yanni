close all;
clear variables;
clc;

c = 91104;
rc = 6955080;
alpha = 1;
g = 9.8;

t1 = -10;
tn = 10;
n = 100000;
dt = (tn - t1)/(n - 1);

t = zeros(1, n);
y = zeros(1, n);
v = zeros(1, n);
a = zeros(1, n);

y1 = zeros(1, n);
v1 = zeros(1, n);
a1 = zeros(1, n);

v(1) = 10;
y(1) = 5;
t(1) = -10;
for i = 1 : n - 1
	t(i+1) = t1 + dt*(i);
    a(i) = g - (k/m)*y(i) - (alpha/m)*v(i);
    v(i+1) = v(i) + a(i)*dt;
    y(i+1) = y(i) + v(i)*dt + a(i)*(dt^2)/2;
end

figure(1)
hold on;
plot(t, y,'-b');
plot(t, v,'-r');
plot(t, a,'-g');
legend('Y coord', 'Speed', 'Accel');
ylabel('Fr = -alphaV', 'FontSize', 14, 'FontWeight','bold')
hold off;

for i = 1 : n - 1
    a1(i) = g - (k/m)*y1(i) - (alpha/m)*v1(i)*abs(v1(i));
    v1(i+1) = v1(i) + a1(i)*dt;
    y1(i+1) = y1(i) + v1(i)*dt + a1(i)*(dt^2)/2;
end

figure(2)
hold on;
plot(t, y1,'-b');
plot(t, v1,'-r');
plot(t, a1,'-g');
legend('Y coord', 'Speed', 'Accel');
ylabel('Fr = -alphaV|V|', 'FontSize', 14, 'FontWeight','bold')
hold off;


%% ლოგისტიკური ფუნქცია
function fnc = logistic(x)
    fnc =  1/(1 + exp(-x));
end
