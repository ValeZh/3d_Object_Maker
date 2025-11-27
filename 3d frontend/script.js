/* ============ DOM элементы ============ */
const shapeDropdown = document.getElementById('shapeDropdown');
const shapeOptionsDiv = document.getElementById('shapeOptions');
const textureDropdown = document.getElementById('textureDropdown');
const textureOptionsDiv = document.getElementById('textureOptions');
const colorInput = document.getElementById('color');
const descriptionInput = document.getElementById('description');
const generateButton = document.getElementById('generate');
const exportButton = document.getElementById('export');
const resetButton = document.getElementById('reset');
const container = document.getElementById('objectPreview');

/* ============ Three.js setup ============ */
let scene, camera, renderer, currentMesh = null;

function initThree() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.set(0, 0, 6);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.innerHTML = '';
  container.appendChild(renderer.domElement);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x222222, 0.6);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(5, 10, 7.5);
  scene.add(dir);
}

/* ============ Load / Update mesh ============ */
async function createMeshFromBackend(shape, colorHex, texture) {
    try {
        const payload = { shape, color: colorHex, texture };
        const res = await fetch('/api/generate-object', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        // Загружаем MTL
        const mtlLoader = new THREE.MTLLoader();
        const mtl = await new Promise((resolve, reject) => {
            mtlLoader.load(data.mtl_url, resolve, undefined, reject);
        });
        mtl.preload();

        // Загружаем OBJ с материалами
        const objLoader = new THREE.OBJLoader();
        objLoader.setMaterials(mtl);
        const obj = await new Promise((resolve, reject) => {
            objLoader.load(data.obj_url, resolve, undefined, reject);
        });

        // Центрирование и масштабирование
        obj.position.set(0, 0, 0);
        const box = new THREE.Box3().setFromObject(obj);
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        obj.scale.setScalar(2 / maxDim);

        return obj;
    } catch (err) {
        console.warn('Error creating mesh from backend:', err);
        return null;
    }
}

async function updateObject(shape, colorHex, texture, textDescription = null) {
  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh.traverse(child => child.geometry?.dispose?.());
  }
  currentMesh = await createMeshFromBackend(shape, colorHex, texture, textDescription);
  if (currentMesh) scene.add(currentMesh);
}

/* ============ Animation loop ============ */
function animate() {
  requestAnimationFrame(animate);
  if (currentMesh) {
    currentMesh.rotation.x += 0.01;
    currentMesh.rotation.y += 0.01;
  }
  renderer.render(scene, camera);
}

/* ============ Dropdown helper ============ */
function setupDropdown(dropdown, optionsDiv, optionsArr, onSelect) {
  optionsDiv.innerHTML = '';
  optionsArr.forEach(opt => {
    const el = document.createElement('div');
    el.textContent = opt.label;
    el.dataset.value = opt.value;
    el.addEventListener('click', () => {
      Array.from(optionsDiv.children).forEach(c => { c.style.background = ''; c.style.color = '#fff'; });
      el.style.background = 'linear-gradient(90deg,#4b00e0,#8e2de2)';
      el.style.color = '#fff';
      dropdown.classList.remove('show');
      onSelect(opt.value, opt.label);
    });
    optionsDiv.appendChild(el);
  });

  const btn = dropdown.querySelector('.dropbtn');
  btn.addEventListener('click', e => {
    e.stopPropagation();
    document.querySelectorAll('.dropdown').forEach(d => { if (d !== dropdown) d.classList.remove('show'); });
    dropdown.classList.toggle('show');
  });
}

window.addEventListener('click', () => document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('show')));

/* ============ Load UI from backend ============ */
let selectedShape = 'Cube';
let selectedTexture = 'stone';

async function loadUI() {
  colorInput.value = '#6952BE';
  try {
    const res = await fetch('/api/options');
    if (!res.ok) throw new Error('Cannot load options');
    const data = await res.json();

    const shapes = data.shapes.map(s => ({ label: s, value: s }));
    const textures = data.textures.map(t => ({ label: t, value: t }));

    setupDropdown(shapeDropdown, shapeOptionsDiv, shapes, (val, label) => {
      selectedShape = val;
      shapeDropdown.querySelector('.dropbtn').textContent = label;
      updateObject(selectedShape, colorInput.value, selectedTexture);
    });

    setupDropdown(textureDropdown, textureOptionsDiv, textures, (val, label) => {
      selectedTexture = val;
      textureDropdown.querySelector('.dropbtn').textContent = label;
      updateObject(selectedShape, colorInput.value, selectedTexture);
    });
  } catch (err) {
    console.warn(err);
  }
}

/* ============ Buttons ============ */
generateButton.addEventListener('click', async () => {
  const desc = descriptionInput.value.trim();
  if (desc.length > 0) {
    await updateObject(null, colorInput.value, null, desc);
  } else {
    await updateObject(selectedShape, colorInput.value, selectedTexture);
  }
});

resetButton.addEventListener('click', () => {
  descriptionInput.value = '';
  selectedShape = 'Cube';
  selectedTexture = 'stone';
  shapeDropdown.querySelector('.dropbtn').textContent = 'Select Shape';
  textureDropdown.querySelector('.dropbtn').textContent = 'Select Texture';
  if (currentMesh) { scene.remove(currentMesh); currentMesh = null; }
});

colorInput.addEventListener('input', () => {
  if (currentMesh) {
    currentMesh.traverse(child => { if (child.isMesh && child.material && child.material.color) child.material.color.set(colorInput.value); });
  }
});

/* ============ Init ============ */
initThree();
loadUI();
updateObject('Cube', '#6952BE', 'stone').then(() => animate());
