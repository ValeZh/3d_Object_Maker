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

/* ============ Geometry helper ============ */
function createGeometryByShape(shape) {
  switch (shape) {
    case 'Cube': return new THREE.BoxGeometry(2, 2, 2);
    case 'Sphere': return new THREE.SphereGeometry(1.2, 64, 64);
    case 'Pyramid': return new THREE.ConeGeometry(1.2, 2.2, 4);
    case 'Prism': return new THREE.CylinderGeometry(1.0, 1.0, 2.0, 6);
    case 'Cylinder': return new THREE.CylinderGeometry(1.0, 1.0, 2.0, 32);
    case 'Cone': return new THREE.ConeGeometry(1.0, 2.0, 32);
    case 'Torus': return new THREE.TorusGeometry(1.0, 0.35, 30, 200);
    default: return new THREE.BoxGeometry(2, 2, 2);
  }
}

/* ============ Create / update mesh ============ */
async function createMeshFromBackend(shape, colorHex, texture) {
  try {
    const payload = { shape, color: colorHex, texture };
    const res = await fetch('/api/generate-object', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    // Загружаем OBJ через OBJLoader
    const objText = await (await fetch(data.obj_url)).text();
    const loader = new THREE.OBJLoader();
    const obj = loader.parse(objText);

    // Применяем текстуру, если есть
    if (data.textures.length > 0) {
      const texUrl = data.textures[0];
      const tmap = new THREE.TextureLoader().load(texUrl);
      obj.traverse(c => {
        if (c.isMesh) c.material = new THREE.MeshStandardMaterial({ map: tmap, color: colorHex });
      });
    } else {
      obj.traverse(c => {
        if (c.isMesh) c.material = new THREE.MeshStandardMaterial({ color: colorHex });
      });
    }

    return obj;
  } catch (err) {
    console.warn('Error creating mesh from backend:', err);
    return null;
  }
}

async function updateObject(shape, colorHex, texture) {
  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh.traverse(child => child.geometry?.dispose?.());
  }
  currentMesh = await createMeshFromBackend(shape, colorHex, texture);
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
function loadUI() {
  colorInput.value = '#6952BE';
  (async () => {
    try {
      const res = await fetch('/api/options');
      if (res.ok) {
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
        return;
      }
    } catch (e) { console.warn(e); }
  })();
}

/* ============ Buttons ============ */
generateButton.addEventListener('click', async () => {
  await updateObject(selectedShape, colorInput.value, selectedTexture);
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
