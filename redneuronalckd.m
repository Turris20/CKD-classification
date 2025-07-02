clc; close all; clear all;

#Cargar datos entrenamiento
bien=csvread('datasetckd_bien_train_limpio.csv');
mal=csvread('datasetckd_mal_train_limpio.csv');


% Posible un ciclo while para entrenar hasta llegar al 80% o superior
error=100
while (error>10);
% dentro del parentesis los primeros : es para
%tomar todos los valores en renglones
%despues de la , es para tomar los valores en columnas
%Variable para determinar el tamaño de nuestros vectores
Dt=31
Bien=bien(1:Dt,1:24); #de 1 a 31 filas, y de 1 a 24 columnas
Mal=mal(1:Dt,1:24);

##################################
function A = softmax(Z)
expZ=exp(Z-max(Z,[],1));
A=expZ ./sum(expZ,1);
end

%proceso de normalizacion por caracteristica
datos = [Bien;Mal];
minVal=min(datos);
maxVal=max(datos);

%mapminmax
ymax=1;
ymin=0.1;
Dato_norm=((ymax-ymin)*(datos-minVal))./(maxVal-minVal)+ymin;

% Entradas para la red neuronal
input = [Dato_norm'];   # el ' esta transponiendo los datos, los gira de columnas a filas


% Targets (salidas deseadas)

T_Bien = [ones(1, Dt); zeros(1, Dt)];
T_Mal = [zeros(1, Dt); ones(1, Dt)];
targets = [T_Bien, T_Mal];


% Hiperparámetros
num_inputs = size(input, 1);
num_hidden = 5; % Número de neuronas en la capa oculta
num_outputs = size(targets, 1);
learning_rate = 0.05;
epochs = 4000;

% Inicialización de pesos y sesgos
W1 = randn(num_hidden, num_inputs);
W2 = randn(num_outputs, num_hidden);

Init_W1=W1;
Init_W2=W2;
%W1 = randn(num_hidden, num_inputs) * 0.1; % Pesos de entrada -> capa oculta
b1 = randn(num_hidden, 1) * 0.1; % Sesgos de la capa oculta
%W2 = randn(num_outputs, num_hidden) * 0.1; % Pesos capa oculta -> salida
b2 = randn(num_outputs, 1) * 0.1; % Sesgos de la capa de salida

Init_b1=b1;
Init_b2=b2;
% Entrenamiento de la red neuronal
for epoch = 1:epochs
% Propagación hacia adelante
Z1 = W1 * input + b1; % Capa oculta (preactivación)
A1 = tanh(Z1); % Activación (tangente hiperbólica)
Z2 = W2 * A1 + b2; % Capa de salida (preactivación)
A2 = softmax(Z2); % Activación (softmax para clasificación)

% Cálculo del error (entropía cruzada)
loss = -sum(sum(targets .* log(A2))) / size(targets, 2);

% Propagación hacia atrás (backpropagation)
dZ2 = A2 - targets; % Error en la salida
dW2 = dZ2 * A1'; % Gradiente de los pesos W2
db2 = sum(dZ2, 2); % Gradiente de los sesgos b2

dA1 = W2' * dZ2; % Error propagado a la capa oculta
dZ1 = dA1 .* (1 - A1.^2); % Gradiente en Z1 (derivada de tanh)
dW1 = dZ1 * input'; % Gradiente de los pesos W1
db1 = sum(dZ1, 2); % Gradiente de los sesgos b1

% Actualización de pesos y sesgos
lambda = 0.05; % Factor de regularización
W1 = W1 - learning_rate * (dW1 + lambda * W1);
W2 = W2 - learning_rate * (dW2 + lambda * W2);


%W1 = W1 - learning_rate * dW1;
b1 = b1 - learning_rate * db1;
%W2 = W2 - learning_rate * dW2;
b2 = b2 - learning_rate * db2;

% Mostrar la pérdida cada 100 épocas
if mod(epoch, 100) == 0
fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
end
end

% Evaluación de la red
Z1 = W1 * input + b1;
A1 = tanh(Z1);
Z2 = W2 * A1 + b2;
A2 = softmax(Z2);

% Predicciones
[~, y_pred] = max(A2, [], 1); % Clases predichas
[~, y_test] = max(targets, [], 1); % Clases reales

% Matriz de confusión
confMat = zeros(num_outputs, num_outputs);
for i = 1:length(y_test)
confMat(y_test(i), y_pred(i)) = confMat(y_test(i), y_pred(i)) + 1;
end
disp('Matriz de confusión:');
disp(confMat);

% Cálculo de métricas
accuracy = sum(diag(confMat)) / sum(confMat(:)) * 100;
error=100-accuracy;
disp(['Precisión: ', num2str(accuracy), '%']);
disp(['Datos de entrenamiento']);
end
% Red creada
red.W1 = W1;
red.b1 = b1;
red.W2 = W2;
red.b2 = b2;

%carga de los datos para evaluacion
Test_bien=csvread('datasetckd_bien_test.csv');
Test_mal=csvread('datasetckd_mal_test.csv');

test_bien=Test_bien(:,1:24); #1 a 24 columnas
test_mal=Test_mal(:,1:24);

%proceso de normalizacion por caracteristica de DATOS TEST
%minVal=min(datos);
%maxVal=max(datos);

ymax=1;
ymin=0.1;
norm_test_bien=((ymax-ymin)*(test_bien-minVal))./(maxVal-minVal)+ymin;
norm_test_mal=((ymax-ymin)*(test_mal-minVal))./(maxVal-minVal)+ymin;

%metricas de evaluacion
TP=0;
TN=0;
FP=0;
FN=0;

for i=1:size(norm_test_bien,1)
  % Propagación hacia adelante
  Z1 = red.W1*norm_test_bien(1,:)'+red.b1; % Capa oculta (preactivación)
  A1 = tanh(Z1); % Activación (tangente hiperbólica)
  Z2 = red.W2 * A1 + red.b2; % Capa de salida (preactivación)
  A2 = softmax(Z2); % Activación (softmax para clasificación)

  %clasificacion
  if (A2(1)>A2(2))
    TP=TP+1;
  else
    FP=FP+1;
  endif

end

for i=1:size(norm_test_mal,1)
  % Propagación hacia adelante
  Z1 = red.W1*norm_test_mal(1,:)'+red.b1; % Capa oculta (preactivación)
  A1 = tanh(Z1); % Activación (tangente hiperbólica)
  Z2 = red.W2 * A1 + red.b2; % Capa de salida (preactivación)
  A2 = softmax(Z2); % Activación (softmax para clasificación)

  %clasificacion
  if (A2(2)>A2(1))
    TN=TN+1;
  else
    FN=FN+1;
  endif

end

Exactitud = ((TP + TN) / (TP+TN+FP+FN))*100
Precision = (TP / (FP + TP))*100
Recall = (TP / (TP+FN))*100
F1 = 2 * (Precision*Recall)/(Precision+Recall)

% Matriz de confusión para prueba
confMat_test = [TP, FP; FN, TN];

disp('Matriz de confusión (prueba):');
disp(confMat_test);

if F1>90
save('red_prueba5.mat','red')
end
