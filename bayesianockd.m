clc; close all; clear all;
pkg load statistics

% Cargar datos de entrenamiento
bien = csvread('datasetckd_bien_train_limpio.csv');
mal = csvread('datasetckd_mal_train_limpio.csv');

% Parámetro
Dt = 31;
Bien = bien(1:Dt,1:24);
Mal = mal(1:Dt,1:24);

% Normalización
datos = [Bien; Mal];
minVal = min(datos);
maxVal = max(datos);
ymax = 1;
ymin = 0.1;

Dato_norm = ((ymax - ymin) * (datos - minVal)) ./ (maxVal - minVal) + ymin;

% Etiquetas: 1 para bien, 0 para mal
etiquetas = [ones(Dt,1); zeros(Dt,1)];

% Dividir datos por clase
X_bien = Dato_norm(etiquetas == 1, :);
X_mal = Dato_norm(etiquetas == 0, :);

% Calcular medias y desviaciones estándar para cada clase
mu_bien = mean(X_bien);
sigma_bien = std(X_bien) + 1e-6; % evitar división por cero

mu_mal = mean(X_mal);
sigma_mal = std(X_mal) + 1e-6;

% Probabilidades a priori
p_bien = size(X_bien, 1) / size(Dato_norm, 1);
p_mal = size(X_mal, 1) / size(Dato_norm, 1);

% Función de predicción usando Naive Bayes Gaussiano
function pred = naive_bayes_predict(X, mu_bien, sigma_bien, mu_mal, sigma_mal, p_bien, p_mal)
  n = size(X,1);
  pred = zeros(n,1);
  for i = 1:n
    x = X(i,:);
    p_x_bien = prod(normpdf(x, mu_bien, sigma_bien)) * p_bien;
    p_x_mal = prod(normpdf(x, mu_mal, sigma_mal)) * p_mal;
    pred(i) = p_x_bien > p_x_mal;
  end
end

% Evaluación en entrenamiento
pred_train = naive_bayes_predict(Dato_norm, mu_bien, sigma_bien, mu_mal, sigma_mal, p_bien, p_mal);

confMat_train = [sum(pred_train==1 & etiquetas==1), sum(pred_train==0 & etiquetas==1);
                 sum(pred_train==1 & etiquetas==0), sum(pred_train==0 & etiquetas==0)];

disp('Matriz de confusión (entrenamiento):');
disp(confMat_train);

accuracy = sum(diag(confMat_train)) / sum(confMat_train(:)) * 100;
disp(['Precisión entrenamiento: ', num2str(accuracy), '%']);

% Cargar datos de prueba
Test_bien = csvread('datasetckd_bien_test.csv');
Test_mal = csvread('datasetckd_mal_test.csv');

test_bien = Test_bien(:,1:24);
test_mal = Test_mal(:,1:24);

% Normalizar
norm_test_bien = ((ymax - ymin)*(test_bien - minVal)) ./ (maxVal - minVal) + ymin;
norm_test_mal = ((ymax - ymin)*(test_mal - minVal)) ./ (maxVal - minVal) + ymin;

% Predicción
pred_bien = naive_bayes_predict(norm_test_bien, mu_bien, sigma_bien, mu_mal, sigma_mal, p_bien, p_mal);
pred_mal = naive_bayes_predict(norm_test_mal, mu_bien, sigma_bien, mu_mal, sigma_mal, p_bien, p_mal);

% Métricas
TP = sum(pred_bien == 1);
FP = sum(pred_bien == 0);
TN = sum(pred_mal == 0);
FN = sum(pred_mal == 1);

% Matriz de confusión para prueba
confMat_test = [TP, FP; FN, TN];

disp('Matriz de confusión (prueba):');
disp(confMat_test);


Exactitud = (TP + TN) / (TP + TN + FP + FN) * 100;
Precision = TP / (TP + FP) * 100;
Recall = TP / (TP + FN) * 100;
F1 = 2 * (Precision * Recall) / (Precision + Recall);

disp(['Exactitud: ', num2str(Exactitud), '%']);
disp(['Precisión: ', num2str(Precision), '%']);
disp(['Recall: ', num2str(Recall), '%']);
disp(['F1 Score: ', num2str(F1), '%']);

if F1 > 90
  save('modelo_nb.mat', 'mu_bien', 'sigma_bien', 'mu_mal', 'sigma_mal', 'p_bien', 'p_mal','minVal','maxVal');
end

