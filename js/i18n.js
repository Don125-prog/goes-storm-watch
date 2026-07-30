/**
 * Internationalization: EN / RU language switcher.
 */

const TRANSLATIONS = {
    en: {
        title: "GOES Storm Watch",
        subtitle: "Binary classification of convective clouds from GOES-16 IR imagery using ResNet-18",
        input: "Input",
        dropText: 'Drag & drop a <code>.npy</code> file here or click to upload',
        dropHint: "Shape: (5, H, W) — 5 IR channels, float32, raw values",
        examplesTitle: "Example crops",
        examplesHint: "Select a region to load, or download and re-upload to test",
        examplesLoading: "Loading examples...",
        examplesNone: "No examples available yet. Run prepare_examples.py first.",
        examplesError: "Examples not available",
        results: "Results",
        overlayOpacity: "Overlay opacity:",
        showGrid: "Show patch grid",
        inputCanvas: "Input (C13 brightness temperature)",
        predCanvas: "Prediction overlay",
        about: "About",
        aboutText: "This tool performs binary classification of convective (cumulonimbus) clouds from GOES-16 ABI infrared satellite imagery. The model is a ResNet-18 trained on 5 IR channels (C07, C09, C13, C14, C15) using combined Cloud Top Phase and brightness temperature labels.",
        techDetails: "Technical details",
        techInput: "5 IR channels, 64×64 patches, stride 48",
        techLabel: "Phase == Ice AND CMI_C13 < 220 K",
        techArch: "ResNet-18, conv1 adapted for 5 channels",
        techPerf: "Precision 91%, Recall 82%, F1 86%, AUC 0.993",
        techInfer: "ONNX Runtime Web (runs entirely in your browser)",
        footer: "Practice project, 2026",
        loadBtn: "Load",
        downloadBtn: "↓",
        // Status messages
        loadingModel: "Loading ONNX model...",
        modelLoaded: "Model loaded",
        modelFailed: "Failed to load model. Make sure resnet18_goes.onnx is in model/",
        uploadNpy: "Please upload a .npy file",
        reading: "Reading",
        invalidShape: "Invalid shape",
        expectedShape: "Expected (5, H, W)",
        errorReading: "Error reading file",
        downloading: "Downloading",
        errorExample: "Error loading example",
        waitingModel: "Waiting for model to load...",
        runningInference: "Running inference...",
        // Stats
        statSize: "Size",
        statPatches: "Patches",
        statConv: "Convective",
        statTime: "Inference",
    },
    ru: {
        title: "GOES Storm Watch",
        subtitle: "Бинарная классификация конвективных облаков по ИК-снимкам GOES-16 с помощью ResNet-18",
        input: "Входные данные",
        dropText: 'Перетащите файл <code>.npy</code> сюда или нажмите для загрузки',
        dropHint: "Формат: (5, H, W) — 5 ИК-каналов, float32, сырые значения",
        examplesTitle: "Примеры кропов",
        examplesHint: "Выберите регион для загрузки или скачайте и загрузите повторно для теста",
        examplesLoading: "Загрузка примеров...",
        examplesNone: "Примеры ещё не готовы. Запустите prepare_examples.py.",
        examplesError: "Примеры недоступны",
        results: "Результаты",
        overlayOpacity: "Прозрачность наложения:",
        showGrid: "Показать сетку патчей",
        inputCanvas: "Вход (яркостная температура C13)",
        predCanvas: "Наложение предсказания",
        about: "О проекте",
        aboutText: "Инструмент выполняет бинарную классификацию конвективных (кучево-дождевых) облаков по инфракрасным спутниковым снимкам GOES-16 ABI. Модель — ResNet-18, обученная на 5 ИК-каналах (C07, C09, C13, C14, C15) с комбинированными метками фазы облачной вершины и яркостной температуры.",
        techDetails: "Технические детали",
        techInput: "5 ИК-каналов, патчи 64×64, шаг 48",
        techLabel: "Phase == Ice AND CMI_C13 < 220 K",
        techArch: "ResNet-18, conv1 адаптирован под 5 каналов",
        techPerf: "Precision 91%, Recall 82%, F1 86%, AUC 0.993",
        techInfer: "ONNX Runtime Web (работает полностью в браузере)",
        footer: "Практический проект, 2026",
        loadBtn: "Загрузить",
        downloadBtn: "↓",
        // Status messages
        loadingModel: "Загрузка модели ONNX...",
        modelLoaded: "Модель загружена",
        modelFailed: "Не удалось загрузить модель. Убедитесь, что resnet18_goes.onnx находится в model/",
        uploadNpy: "Пожалуйста, загрузите файл .npy",
        reading: "Чтение",
        invalidShape: "Неверная форма",
        expectedShape: "Ожидается (5, H, W)",
        errorReading: "Ошибка чтения файла",
        downloading: "Скачивание",
        errorExample: "Ошибка загрузки примера",
        waitingModel: "Ожидание загрузки модели...",
        runningInference: "Выполнение инференса...",
        // Stats
        statSize: "Размер",
        statPatches: "Патчи",
        statConv: "Конвективные",
        statTime: "Инференс",
    }
};

let currentLang = localStorage.getItem('gsw-lang') || 'en';

function t(key) {
    return TRANSLATIONS[currentLang][key] || TRANSLATIONS.en[key] || key;
}

function setLang(lang) {
    currentLang = lang;
    localStorage.setItem('gsw-lang', lang);
    applyTranslations();
}

function applyTranslations() {
    // Header
    document.getElementById('site-title').textContent = t('title');
    document.getElementById('site-subtitle').textContent = t('subtitle');

    // Input panel
    document.getElementById('input-heading').textContent = t('input');
    document.getElementById('drop-text').innerHTML = t('dropText');
    document.getElementById('drop-hint').textContent = t('dropHint');
    document.getElementById('examples-heading').textContent = t('examplesTitle');
    document.getElementById('examples-hint').textContent = t('examplesHint');

    // Results panel
    document.getElementById('results-heading').textContent = t('results');
    document.getElementById('label-opacity').firstChild.textContent = t('overlayOpacity') + ' ';
    document.getElementById('label-grid-text').textContent = ' ' + t('showGrid');
    document.getElementById('input-canvas-title').textContent = t('inputCanvas');
    document.getElementById('pred-canvas-title').textContent = t('predCanvas');

    // About
    document.getElementById('about-heading').textContent = t('about');
    document.getElementById('about-text').textContent = t('aboutText');
    document.getElementById('tech-summary').textContent = t('techDetails');
    document.getElementById('tech-input').innerHTML = `<strong>Input:</strong> ${t('techInput')}`;
    document.getElementById('tech-label').innerHTML = `<strong>Label:</strong> ${t('techLabel')}`;
    document.getElementById('tech-arch').innerHTML = `<strong>Architecture:</strong> ${t('techArch')}`;
    document.getElementById('tech-perf').innerHTML = `<strong>Performance:</strong> ${t('techPerf')}`;
    document.getElementById('tech-infer').innerHTML = `<strong>Inference:</strong> ${t('techInfer')}`;

    // Footer
    document.getElementById('footer-text').textContent = t('footer');

    // Lang switcher active state
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.lang === currentLang);
    });

    // Re-render examples with translated buttons
    if (typeof reloadExamplesUI === 'function') reloadExamplesUI();

    // Re-render stats if results are showing
    if (typeof currentResult !== 'undefined' && currentResult) {
        updateResultsStats();
    }
}
