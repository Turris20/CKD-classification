clc; close all; clear all;

function interfaz_ckd_dual_predictor()
  pkg load statistics

  % === CARGA DE MODELOS ENTRENADOS ===
  load('modelo_nb.mat');       % Modelo Bayesiano
  load('red_prueba4.mat');     % Red neuronal
  load('modelo_svm.mat');      % Modelo SVM (contiene 'theta')

  % Lista de nombres de características
  nombres = {
    'age', 'blood pressure(bp)', 'specific gravity(sg)', 'albumin(al)', 'sugar(su)', 'red blood cells(rbc)', 'pus cell(pc)', 'pus cell clumps(pcc)', 'bacteria(ba)', ...
    'blood glucose random(bgr)', 'blood urea(bu)', 'serum creatinine(sc)', 'sodium(sod)', 'potassium(pot)', 'hemoglobin(hemo)', 'packed cell volume(pcv)', 'white blood cell count(wc)', ...
    'red blood cell count(rc)', 'hypertension(htn)', 'diabetes mellitus(dm)', 'coronary artery disease(cad)', 'appetite(appet)', 'pedal edema(pe)', 'anemia(ane)'
  };
  mensajes_ayuda = {
  'Edad en años del paciente',
  'Presión arterial median en mmHg',
  'nominal: 1.005, 1.010, 1.015, 1.020 o 1.025'
  'nominal: 0, 1, 2, 3, 4 o 5',
  'nominal: 0, 1, 2, 3, 4 o 5',
  "anormal: 0 \nnormal: 1",
  "anormal: 0 \nnormal: 1",
  "no presenta: 0 \npresenta: 1",
  "no presenta: 0 \npresenta: 1",
  "Medición aleatoria de glucosa \nen sangre en mgs/dl",
  'Urea sanguínea en mgs/dl',
  "Creatinina sérica en mgs/dl",
  "Concentración de sodio en mEq/L",
  "Concentración de potasio en mEq/L",
  "Hemoglobina en gms",
  "Hematocrito numérico",
  "Conteo de glóbulos blancos en células/mm3",
  "Conteo de glóbulos rojos en millones/mm3",
  "no: 0 \nsi:1",
  "no: 0 \nsi:1",
  "no: 0 \nsi:1",
  "pobre: 0 \nbueno: 1",
  "no: 0 \nsi:1",
  "no: 0 \nsi:1",
  };

  % Crear ventana
  f = figure('Name', 'Chronic kidney disease calculator', ...
             'Position', [200, 0, 800, 700]);
  uicontrol(f,'Style','text','String', 'Chronic kidney disease calculator',...
            'Position',[20,650,800,20],...
            'HorizontalAlignment', 'center', 'FontSize', 14);
  % Crear campos de entrada
  campos = cell(1, 24);
  for i = 1:24
  if i <= 12
    xLabel = 50;
    xInput = 220;
    y = 650 - i*35;
  else
    xLabel = 380;
    xInput = 620;
    y = 650 - (i-12)*35;
  end

  uicontrol(f, 'Style', 'text', 'String', nombres{i}, ...
            'Position', [xLabel, y, 210, 25], ...
            'HorizontalAlignment', 'left', 'FontSize', 10);

  campos{i} = uicontrol(f, 'Style', 'edit', ...
                        'Position', [xInput, y, 80, 25], ...
                        'FontSize', 10);

  % Botón de ayuda al lado derecho del campo
  ayuda_msg = [mensajes_ayuda{i};];  % Aquí podrías personalizar cada mensaje
  uicontrol(f, 'Style', 'pushbutton', 'String', 'ⓘ', ...
            'Position', [xInput + 90, y, 25, 25], ...
            'TooltipString', 'Haz clic para ayuda', ...
            'Callback', @(src, event) msgbox(ayuda_msg));
end

  % Área para mostrar resultados
  % Panel gris que simula recuadro
  panel_resultado = uipanel(f, 'Position', [0.12 0.05 0.76 0.13], ... % Relativo al figure
                          'BackgroundColor', [0.8 0.8 0.8], ...
                          'BorderType', 'line', ...
                          'Title', 'Results');

  % Texto dentro del panel
  resultado_texto = uicontrol(panel_resultado, 'Style', 'text', 'String', '', ...
                            'Units', 'normalized', ...
                            'Position', [0.05 0.1 0.9 0.8], ...
                            'FontSize', 11, ...
                            'HorizontalAlignment', 'center', ...
                            'BackgroundColor', [0.8 0.8 0.8]);


  % Botón Calcular
  uicontrol(f, 'Style', 'pushbutton', 'String', 'Calculate', ...
            'Position', [300, 150, 100, 35], ...
            'FontSize', 11, ...
            'Callback', @(src, event) calcularTodosLosModelos(campos, ...
                        mu_bien, sigma_bien, mu_mal, sigma_mal, ...
                        p_bien, p_mal, minVal, maxVal, ...
                        red, theta, resultado_texto));
end

function calcularTodosLosModelos(campos, mu_bien, sigma_bien, mu_mal, sigma_mal, ...
                                 p_bien, p_mal, minVal, maxVal, ...
                                 red, theta, resultado_texto)
  % === Lectura de datos ===
  entrada = zeros(1, 24);
  for i = 1:24
    entrada(i) = str2double(get(campos{i}, 'String'));
  end

  % Validación
  if any(isnan(entrada))
    set(resultado_texto, 'String', '⚠️ Error: Verifica que todos los campos estén llenos con números.');
    return;
  end

  % === Normalización para Bayes y Red ===
  ymin = 0.1;
  ymax = 1;
  entrada_norm = ((ymax - ymin)*(entrada - minVal)) ./ (maxVal - minVal) + ymin;

  % === Modelo Bayesiano ===
  p_x_bien = prod(normpdf(entrada_norm, mu_bien, sigma_bien)) * p_bien;
  p_x_mal  = prod(normpdf(entrada_norm, mu_mal, sigma_mal)) * p_mal;

  if p_x_bien > p_x_mal
    resultado_bayes = "✅ (Bayesiano): The patient doesn't have CKD.";
  else
    resultado_bayes = '⚠️ (Bayesiano): The patient have CKD.';
  end

  % === Red Neuronal ===
  Z1 = red.W1 * entrada_norm' + red.b1;
  A1 = tanh(Z1);
  Z2 = red.W2 * A1 + red.b2;
  A2 = softmax(Z2);

  if A2(1) > A2(2)
    resultado_nn = "✅ (Red Neuronal): The patient doesn't have CKD.";
  else
    resultado_nn = '⚠️ (Red Neuronal): The patient have CKD.';
  end

  % === Modelo SVM ===
  hemo = entrada(15);
  pcv  = entrada(16);
  x_svm = [1, hemo, pcv];  % Agregar bias

  prob_svm = sigmoid(x_svm * theta);
  if prob_svm >= 0.5
    resultado_svm = '⚠️ (SVM): The patient have CKD.';
  else
    resultado_svm = "✅ (SVM): The patient doesn't have CKD.";
  end

  % === Mostrar resultados finales ===
  set(resultado_texto, 'String', sprintf('%s\n%s\n%s', ...
      resultado_bayes, resultado_nn, resultado_svm));
end

function A = softmax(Z)
  expZ = exp(Z - max(Z, [], 1));  % para evitar overflow numérico
  A = expZ ./ sum(expZ, 1);
end

function g = sigmoid(z)
  g = 1.0 ./ (1.0 + exp(-z));
end

% Ejecutar interfaz
interfaz_ckd_dual_predictor();

