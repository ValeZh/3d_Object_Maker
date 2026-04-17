/* script.js — адаптирован для House Builder интерфейса */

const SERVER_URL = location.origin;

// ======================= PAGE TABS =======================

const tabModules = document.getElementById('tabModules');
const tabLibrary = document.getElementById('tabLibrary');
const tabBuilder = document.getElementById('tabBuilder');

const modulePage = document.getElementById('modulePage');
const libraryPage = document.getElementById('libraryPage');
const builderPage = document.getElementById('builderPage');

// ======================= THREE.JS SETUP =======================

let scene, camera, renderer, controls, currentMesh = null;
let blobUrls = [];

async function waitForThree() {
  return new Promise(resolve => {
    const check = () => {
      if (window.THREE && THREE.OrbitControls && THREE.OBJLoader && THREE.MTLLoader) {
        resolve();
      } else {
        setTimeout(check, 50);
      }
    };
    check();
  });
}

async function initScene() {
  await waitForThree();

  const preview = document.getElementById('objectPreview');

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x121212);

  camera = new THREE.PerspectiveCamera(
    60,
    preview.clientWidth / preview.clientHeight,
    0.1,
    200
  );
  camera.position.set(0, 1, 6);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(preview.clientWidth, preview.clientHeight);
  renderer.setPixelRatio(devicePixelRatio);
  preview.appendChild(renderer.domElement);

  const light = new THREE.HemisphereLight(0xffffff, 0x333333, 1.5);
  scene.add(light);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  window.addEventListener('resize', () => {
    camera.aspect = preview.clientWidth / preview.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(preview.clientWidth, preview.clientHeight);
  });

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// ======================= TAB SWITCHING =======================

function switchTab(pageId) {
  modulePage.classList.add('hidden');
  libraryPage.classList.add('hidden');
  builderPage.classList.add('hidden');

  tabModules.classList.remove('active-tab');
  tabLibrary.classList.remove('active-tab');
  tabBuilder.classList.remove('active-tab');

  if (pageId === 'module') {
    modulePage.classList.remove('hidden');
    tabModules.classList.add('active-tab');
  } else if (pageId === 'library') {
    libraryPage.classList.remove('hidden');
    tabLibrary.classList.add('active-tab');
  } else if (pageId === 'builder') {
    builderPage.classList.remove('hidden');
    tabBuilder.classList.add('active-tab');
  }
}

tabModules.onclick = () => switchTab('module');
tabLibrary.onclick = () => switchTab('library');
tabBuilder.onclick = () => switchTab('builder');

// ======================= HOUSE BUILDER LOGIC =======================

const houseDescription = document.getElementById('houseDescription');
const floorsSlider = document.getElementById('floors');
const floorsValue = document.getElementById('floorsValue');
const generateHouseBtn = document.getElementById('generateHouse');
const analyzeHouseBtn = document.getElementById('analyzeHouse');
const previewJson = document.getElementById('previewJson');

// Update slider values
document.getElementById('floors').oninput = () => {
  document.getElementById('floorsValue').textContent = this.value;
};
document.getElementById('sections').oninput = () => {
  document.getElementById('sectionsValue').textContent = this.value;
};
document.getElementById('width').oninput = () => {
  document.getElementById('widthValue').textContent = this.value;
};
document.getElementById('depth').oninput = () => {
  document.getElementById('depthValue').textContent = this.value;
};
document.getElementById('balconyRate').oninput = () => {
  document.getElementById('balconyRateValue').textContent = this.value;
};
document.getElementById('windowCols').oninput = () => {
  document.getElementById('windowColsValue').textContent = this.value;
};

// ======================= ANALYZE HOUSE (DeepSeek) =======================

analyzeHouseBtn.onclick = async () => {
  const text = houseDescription.value.trim();

  if (!text) {
    alert('Пожалуйста, опиши дом!');
    return;
  }

  analyzeHouseBtn.disabled = true;
  analyzeHouseBtn.textContent = 'Analyzing...';

  try {
    console.log('📤 Отправляю текст на сервер:', text);

    const response = await fetch(`${SERVER_URL}/api/generate-building`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    });

    console.log('📥 Получен ответ:', response.status);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Server error');
    }

    const data = await response.json();
    console.log('✅ Результат:', data);

    // Показываем извлеченные параметры
    previewJson.textContent = JSON.stringify(data.parameters, null, 2);

    // Загружаем ZIP если есть
    if (data.zip_url) {
      console.log('📦 Загружаю ZIP:', data.zip_url);
      await loadZip(data.zip_url);
    }

  } catch (error) {
    console.error('❌ Ошибка:', error);
    alert('Ошибка: ' + error.message);
  } finally {
    analyzeHouseBtn.disabled = false;
    analyzeHouseBtn.textContent = 'Analyze';
  }
};

// ======================= GENERATE HOUSE (FROM SLIDERS) =======================

generateHouseBtn.onclick = async () => {
  const floors = parseInt(document.getElementById('floors').value);
  const sections = parseInt(document.getElementById('sections').value);
  const width = parseInt(document.getElementById('width').value);
  const depth = parseInt(document.getElementById('depth').value);
  const hasBalconies = document.getElementById('hasBalconies').checked;
  const balconyRate = parseFloat(document.getElementById('balconyRate').value);

  // Создаем текстовое описание из параметров
  const text = `${floors} этажей, ${width} метров длины, ${depth} метров глубины, ${sections} входов${hasBalconies ? ', с балконами' : ''}`;

  console.log('📝 Генерирую дом из параметров:', { floors, sections, width, depth, hasBalconies });
  console.log('📤 Отправляю текст:', text);

  generateHouseBtn.disabled = true;
  generateHouseBtn.textContent = 'Generating...';

  try {
    const response = await fetch(`${SERVER_URL}/api/generate-building`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Server error');
    }

    const data = await response.json();
    console.log('✅ Дом сгенерирован:', data);

    // Показываем параметры
    previewJson.textContent = JSON.stringify(data.parameters, null, 2);

    // Загружаем ZIP
    if (data.zip_url) {
      await loadZip(data.zip_url);
    }

  } catch (error) {
    console.error('❌ Ошибка генерации:', error);
    alert('Ошибка: ' + error.message);
  } finally {
    generateHouseBtn.disabled = false;
    generateHouseBtn.textContent = 'Generate House';
  }
};

// ======================= ZIP LOADING ==================

async function loadZip(url) {
  try {
    console.log('🔍 Загружаю ZIP с URL:', url);

    if (url.startsWith('/')) {
      url = SERVER_URL + url;
    }

    blobUrls.forEach(u => URL.revokeObjectURL(u));
    blobUrls = [];

    const res = await fetch(url);

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }

    const arrayBuffer = await res.arrayBuffer();
    console.log('📦 ZIP загружен, размер:', arrayBuffer.byteLength, 'bytes');

    const zip = await JSZip.loadAsync(arrayBuffer);
    console.log('📂 Файлы в ZIP:', Object.keys(zip.files));

    let objText = null;
    let mtlText = null;
    const textureMap = {};

    // Ищем OBJ, MTL и текстуры
    for (const path in zip.files) {
      const file = zip.files[path];

      if (path.endsWith('.obj')) {
        console.log('📄 Найден OBJ:', path);
        objText = await file.async('text');
      }

      if (path.endsWith('.mtl')) {
        console.log('📄 Найден MTL:', path);
        mtlText = await file.async('text');
      }

      if (/\.(png|jpg|jpeg)$/i.test(path)) {
        console.log('🖼️ Найдена текстура:', path);
        const blob = await file.async('blob');
        const blobUrl = URL.createObjectURL(blob);
        textureMap[path.split('/').pop()] = blobUrl;
        blobUrls.push(blobUrl);
      }
    }

    if (!objText) {
      throw new Error('OBJ файл не найден в ZIP');
    }

    console.log('✅ Файлы загружены');

    // Загружаем модель в Three.js
    if (currentMesh) {
      scene.remove(currentMesh);
    }

    const materials = mtlText ? new THREE.MTLLoader().parse(mtlText) : null;
    const loader = new THREE.OBJLoader();
    if (materials) {
        try {
        loader.setMaterials(materials);
    } catch (e) {
        console.warn('⚠️ Ошибка загрузки материалов, используем дефолт:', e);
        // Игнорируем ошибку и загружаем без материалов
        }
    }

    const model = loader.parse(objText);
    console.log('🎨 Модель загружена, сетки:', model.children.length);

    // Центрируем и масштабируем
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    model.position.sub(center);

    const size = box.getSize(new THREE.Vector3()).length();
    if (size > 0) {
      model.scale.setScalar(2.6 / size);
    }

    currentMesh = model;
    scene.add(model);

    console.log('✅ Модель отображена в сцене');

  } catch (error) {
    console.error('❌ Ошибка загрузки ZIP:', error);
    alert('Ошибка загрузки модели: ' + error.message);
  }
}

// ======================= MODULE GENERATOR ==================

const analyzeModuleBtn = document.getElementById('analyzeModule');
const moduleJson = document.getElementById('moduleJson');

analyzeModuleBtn.onclick = async () => {
  const moduleDescription = document.getElementById('moduleDescription').value.trim();

  if (!moduleDescription) {
    alert('Пожалуйста, опиши модуль!');
    return;
  }

  analyzeModuleBtn.disabled = true;
  analyzeModuleBtn.textContent = 'Analyzing...';

  try {
    console.log('📤 Анализирую модуль:', moduleDescription);

    const response = await fetch(`${SERVER_URL}/api/generate-building`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: moduleDescription })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Server error');
    }

    const data = await response.json();
    console.log('✅ Анализ завершен:', data);

    moduleJson.textContent = JSON.stringify(data.parameters, null, 2);

  } catch (error) {
    console.error('❌ Ошибка:', error);
    alert('Ошибка: ' + error.message);
  } finally {
    analyzeModuleBtn.disabled = false;
    analyzeModuleBtn.textContent = 'Analyze';
  }
};

// ======================= INITIALIZATION ==================

(async () => {
  console.log('🚀 Инициализация приложения...');
  await initScene();
  switchTab('builder'); // По умолчанию открываем House Builder
  console.log('✅ Готово!');
})();

// ======================= RESET BUTTON ==================

document.getElementById('reset').onclick = () => {
  location.reload();
};