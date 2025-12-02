/* script.js — 100% рабочий, декабрь 2025 */
const SERVER_URL = 'http://127.0.0.1:8000';

const shapeDropdown = document.getElementById('shapeDropdown');
const shapeBtn = shapeDropdown.querySelector('.dropbtn');
const shapeOptions = document.getElementById('shapeOptions');

const textureDropdown = document.getElementById('textureDropdown');
const textureBtn = textureDropdown.querySelector('.dropbtn');
const textureOptions = document.getElementById('textureOptions');

const colorInput = document.getElementById('color');
const descriptionInput = document.getElementById('description');
const generateBtn = document.getElementById('generate');
const exportBtn = document.getElementById('export');
const resetBtn = document.getElementById('reset');
const preview = document.getElementById('objectPreview');

let scene, camera, renderer, controls, currentMesh = null;
let blobUrls = [];

// Ждём загрузки Three.js
function waitForThree() {
  return new Promise(resolve => {
    const check = () => {
      if (window.THREE && THREE.OrbitControls && THREE.OBJLoader) resolve();
      else setTimeout(check, 50);
    };
    check();
  });
}

async function init() {
  await waitForThree();

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  camera = new THREE.PerspectiveCamera(60, preview.clientWidth / preview.clientHeight, 0.1, 1000);
  camera.position.set(0, 0, 5);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(preview.clientWidth, preview.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  preview.appendChild(renderer.domElement);

  const light = new THREE.HemisphereLight(0xffffff, 0x444444, 1.5);
  scene.add(light);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  window.addEventListener('resize', () => {
    camera.aspect = preview.clientWidth / preview.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(preview.clientWidth, preview.clientHeight);
  });

  loadOptions();
  animate();
}

function loadOptions() {
  fetch(SERVER_URL + '/api/options')
    .then(r => r.json())
    .then(data => {
      populate(shapeOptions, data.shapes || ["cube", "sphere", "cylinder"]);
      populate(textureOptions, data.textures || ["wood", "stone", "metallic"]);
    })
    .catch(() => {
      populate(shapeOptions, ["cube", "sphere", "cylinder"]);
      populate(textureOptions, ["wood", "stone"]);
    });
}

function populate(container, items) {
  container.innerHTML = '';
  items.forEach(item => {
    const div = document.createElement('div');
    div.textContent = item;
    div.onclick = () => {
      if (container === shapeOptions) {
        shapeBtn.textContent = item;
      } else {
        textureBtn.textContent = item;
      }
      container.parentElement.classList.remove('show');
    };
    container.appendChild(div);
  });
}

// Закрытие дропдаунов
document.addEventListener('click', e => {
  document.querySelectorAll('.dropdown').forEach(d => {
    if (!d.contains(e.target)) d.classList.remove('show');
  });
});
shapeBtn.onclick = e => { e.stopPropagation(); shapeDropdown.classList.toggle('show'); };
textureBtn.onclick = e => { e.stopPropagation(); textureDropdown.classList.toggle('show'); };

generateBtn.onclick = async () => {
  const text = descriptionInput.value.trim();
  const payload = text ? { text } : {
    shape: shapeBtn.textContent.includes('Выберите') ? 'cube' : shapeBtn.textContent,
    texture: textureBtn.textContent.includes('Выберите') ? 'wood' : textureBtn.textContent,
    color: colorInput.value
  };

  generateBtn.disabled = true;
  generateBtn.textContent = 'Генерация...';

  try {
    const endpoint = text ? '/api/generate-from-text' : '/api/generate-object';
    const res = await fetch(SERVER_URL + endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.zip_url) await loadZip(data.zip_url);
  } catch (err) {
    alert('Ошибка: ' + err.message);
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = 'Generate';
  }
};

async function loadZip(url) {
  if (url.startsWith('/')) url = SERVER_URL + url;
  blobUrls.forEach(u => URL.revokeObjectURL(u));
  blobUrls = [];

  const res = await fetch(url);
  const zip = await JSZip.loadAsync(await res.arrayBuffer());

  let objText = null, mtlText = null;
  const textures = {};

  for (const path in zip.files) {
    const file = zip.files[path];
    if (path.endsWith('.obj')) objText = await file.async('text');
    if (path.endsWith('.mtl')) mtlText = await file.async('text');
    if (/\.(jpe?g|png)$/i.test(path)) {
      const blob = await file.async('blob');
      const blobUrl = URL.createObjectURL(blob);
      textures[path.split('/').pop()] = blobUrl;
      blobUrls.push(blobUrl);
    }
  }

  let materials = null;
  if (mtlText) {
    for (const [name, url] of Object.entries(textures)) {
      mtlText = mtlText.replace(new RegExp(name, 'g'), url);
    }
    materials = new THREE.MTLLoader().parse(mtlText);
  }

  const obj = new THREE.OBJLoader();
  if (materials) obj.setMaterials(materials);
  const model = obj.parse(objText);

  if (currentMesh) scene.remove(currentMesh);
  currentMesh = model;
  scene.add(currentMesh);

  // Центрирование
  const box = new THREE.Box3().setFromObject(model);
  const center = box.getCenter(new THREE.Vector3());
  model.position.sub(center);
  const size = box.getSize(new THREE.Vector3()).length();
  model.scale.setScalar(3 / size);
}

exportBtn.onclick = () => {
  if (!currentMesh) return alert("Сначала сгенерируйте объект");
  const exporter = new THREE.OBJExporter();
  const obj = exporter.parse(currentMesh);
  const blob = new Blob([obj], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'model.obj';
  a.click();
};

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

resetBtn.onclick = () => location.reload();

init();