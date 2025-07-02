clc; clear all; close all;
pkg load io;
pkg load optim;

function [J, grad] = costFunction(theta, X, y)
  m = length(y); % número de ejemplos

  % Hipótesis sigmoide
  h = sigmoid(X * theta);

  % Costo (función logística)
  J = (1/m) * (-y' * log(h) - (1 - y)' * log(1 - h));

  % Gradiente
  grad = (1/m) * X' * (h - y);
end

function g = sigmoid(z)
  g = 1.0 ./ (1.0 + exp(-z));
end


% === CARGA DE DATOS DE ENTRENAMIENTO ===
data_sano = csvread('datasetckd_bien_train_limpio.csv');
data_enfermo = csvread('datasetckd_mal_train_limpio.csv');

X_sano = data_sano(:, 15:16);
X_enfermo = data_enfermo(:, 15:16);

y_sano = zeros(size(X_sano, 1), 1);     % Clase 0: sano
y_enfermo = ones(size(X_enfermo, 1), 1); % Clase 1: enfermo

X = [X_sano; X_enfermo];
y = [y_sano; y_enfermo];

% === AGREGAR INTERCEPTO ===
X_aug = [ones(size(X, 1), 1), X];

% === ENTRENAMIENTO ===
initial_theta = zeros(size(X_aug, 2), 1);
theta = fminunc(@(t)(costFunction(t, X_aug, y)), initial_theta);

Wb = theta(1);
Wx = theta(2);
Wy = theta(3);

% === GRAFICAR DATOS ENTRENAMIENTO ===
figure;
gscatter(X(:,1), X(:,2), y);
xlabel('Hemoglobin (hemo)');
ylabel('Packed cell volume (pcv)');
title('Train');
hold on;

% Línea de decisión
x_vals = linspace(min(X(:,1)), max(X(:,1)), 100);
y_vals = -(Wx / Wy) * x_vals - (Wb / Wy);
plot(x_vals, y_vals, 'm-', 'LineWidth', 2, 'DisplayName', 'Decision boundary');


% === 3 PARES MÁS CERCANOS ENTRE CLASES DIFERENTES ===
n = size(X, 1);
pares = [];

for i = 1:n-1
  for j = i+1:n
    if y(i) != y(j)
      d = norm(X(i,:) - X(j,:));
      pares = [pares; i, j, d];
    end
  end
end

pares_ordenados = sortrows(pares, 3);

disp('=== 3 PARES MÁS CERCANOS ENTRE CLASES DIFERENTES ===');
for k = 1:3
  i = pares_ordenados(k, 1);
  j = pares_ordenados(k, 2);
  d = pares_ordenados(k, 3);
  xi = X(i, :);
  xj = X(j, :);

  printf("\nPar %d:\n", k);
  printf("  Punto %d (Clase %d): (%.4f, %.4f)\n", i, y(i), xi(1), xi(2));
  printf("  Punto %d (Clase %d): (%.4f, %.4f)\n", j, y(j), xj(1), xj(2));
  printf("  Distancia: %.4f\n", d);

  plot([xi(1), xj(1)], [xi(2), xj(2)], 'k--', 'LineWidth', 1, 'HandleVisibility','off');

end

% === CLASIFICACIÓN DE DATOS DE TEST ===
data_sano_test = csvread('datasetckd_bien_test.csv');
data_enfermo_test = csvread('datasetckd_mal_test.csv');

X_test_sano = data_sano_test(:, 15:16);
X_test_enfermo = data_enfermo_test(:, 15:16);

y_test_sano = zeros(size(X_test_sano, 1), 1);
y_test_enfermo = ones(size(X_test_enfermo, 1), 1);

X_test = [X_test_sano; X_test_enfermo];
y_real = [y_test_sano; y_test_enfermo];

num_puntos = size(X_test, 1);
predicciones = zeros(num_puntos, 1);

for i = 1:num_puntos
    x = X_test(i, 1);
    y_ = X_test(i, 2);
    resultado = Wx * x + Wy * y_ + Wb;
    predicciones(i) = resultado >= 0;  % Umbral en 0
end

% Comparar con etiquetas reales
aciertos = sum(predicciones == y_real);
errores = num_puntos - aciertos;
porcentaje = (aciertos / num_puntos) * 100;

% Mostrar resultados
disp('=== RESULTADOS EN TEST ===');
disp('Predicciones:');
disp(predicciones);
disp('Etiquetas reales:');
disp(y_real);
fprintf('✔ Aciertos: %d\n', aciertos);
fprintf('✘ Errores: %d\n', errores);
fprintf('🎯 Porcentaje de precisión: %.2f%%\n', porcentaje);

% === MÉTRICAS DE EVALUACIÓN ===
TP = sum((predicciones == 1) & (y_real == 1));
TN = sum((predicciones == 0) & (y_real == 0));
FP = sum((predicciones == 1) & (y_real == 0));
FN = sum((predicciones == 0) & (y_real == 1));

% Matriz de confusión
confMat_test = [TP, FP; FN, TN];

disp('Matriz de confusión (prueba):');
disp(confMat_test);

% Métricas
Exactitud = (TP + TN) / (TP + TN + FP + FN) * 100;
Precision = TP / (TP + FP + eps) * 100;
Recall = TP / (TP + FN + eps) * 100;
F1 = 2 * (Precision * Recall) / (Precision + Recall + eps);

disp(['Exactitud: ', num2str(Exactitud), '%']);
disp(['Precisión: ', num2str(Precision), '%']);
disp(['Recall: ', num2str(Recall), '%']);
disp(['F1 Score: ', num2str(F1), '%']);

% Guardar modelo si F1 > 90%
if F1 > 90
  save('modelo_logistico.mat', 'theta');
end


save('modelo_svm.mat', 'theta');


