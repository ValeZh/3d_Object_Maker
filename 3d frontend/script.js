/* script.js — стабильная версия */

const SERVER_URL = location.origin; // <-- Работает без CORS

// UI
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

// ================= THREE =================

async function waitForThree() {
  return new Promise(resolve => {
    const check = () => {
      if (window.THREE && THREE.OrbitControls && THREE.OBJLoader && THREE.MTLLoader) resolve();
      else setTimeout(check, 50);
    };
    check();
  });
}

async function initScene() {
  await waitForThree();

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x121212);

  camera = new THREE.PerspectiveCamera(60, preview.clientWidth / preview.clientHeight, 0.1, 200);
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

// ================= UI LOAD OPTIONS =================

async function loadOptions() {
  try {
    const r = await fetch(`${SERVER_URL}/api/options`);
    const data = await r.json();
    populate(shapeOptions, data.shapes);
    populate(textureOptions, data.textures);
  } catch (e) {
    populate(shapeOptions, ["cube", "sphere", "cylinder"]);
    populate(textureOptions, ["wood", "metallic"]);
  }
}

function populate(container, arr) {
  container.innerHTML = "";
  for (const value of arr) {
    const div = document.createElement('div');
    div.className = "dropdown-item";
    div.textContent = value;
    div.onclick = () => {
      (container === shapeOptions ? shapeBtn : textureBtn).textContent = value;
      container.parentElement.classList.remove("show");
    };
    container.appendChild(div);
  }
}

// закрытие меню
document.addEventListener('click', e => {
  document.querySelectorAll(".dropdown").forEach(d => {
    if (!d.contains(e.target)) d.classList.remove("show");
  });
});

// открытие
shapeBtn.onclick = e => { e.stopPropagation(); shapeDropdown.classList.toggle("show"); };
textureBtn.onclick = e => { e.stopPropagation(); textureDropdown.classList.toggle("show"); };

// ================= GENERATION =================

generateBtn.onclick = async () => {
  const desc = descriptionInput.value.trim();
  const payload = desc
    ? { text: desc }
    : {
        shape: shapeBtn.textContent || "cube",
        texture: textureBtn.textContent || "wood",
        color: colorInput.value,
      };

  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";

  try {
    const api = desc ? `/api/generate-from-text` : `/api/generate-object`;
    const r = await fetch(`${SERVER_URL}${api}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await r.json();
    if (!data.zip_url) throw Error("No zip returned");

    await loadZip(data.zip_url);
  } catch (err) {
    alert("Generation failed: " + err.message);
  }

  generateBtn.disabled = false;
  generateBtn.textContent = "Generate";
};


// ================= ZIP PARSING =================

async function loadZip(url) {
  if (url.startsWith("/")) url = SERVER_URL + url;
  blobUrls.forEach(u => URL.revokeObjectURL(u));
  blobUrls = [];

  const res = await fetch(url);
  const zip = await JSZip.loadAsync(await res.arrayBuffer());

  let objText = null;
  let mtlText = null;
  const textureMap = {};

  for (const p in zip.files) {
    const f = zip.files[p];
    if (p.endsWith(".obj")) objText = await f.async("text");
    if (p.endsWith(".mtl")) mtlText = await f.async("text");
    if (/\.(png|jpg|jpeg)$/i.test(p)) {
      const blob = await f.async("blob");
      const u = URL.createObjectURL(blob);
      textureMap[p.split("/").pop()] = u;
      blobUrls.push(u);
    }
  }

  // корректная подмена пути к текстуре
  if (mtlText) {
    for (const texName in textureMap) {
      mtlText = mtlText.replace(
        new RegExp(`map_Kd\\s+.*${texName}`, "g"),
        `map_Kd ${textureMap[texName]}`
      );
    }
  }

  const materials = mtlText ? new THREE.MTLLoader().parse(mtlText) : null;
  const obj = new THREE.OBJLoader();
  if (materials) obj.setMaterials(materials);

  const model = obj.parse(objText);

  if (currentMesh) scene.remove(currentMesh);

  const box = new THREE.Box3().setFromObject(model);
  const center = box.getCenter(new THREE.Vector3());
  model.position.sub(center);

  const size = box.getSize(new THREE.Vector3()).length();
  model.scale.setScalar(2.6 / size);

  currentMesh = model;
  scene.add(model);
}

// ================= EXPORT =================

exportBtn.onclick = () => {
  if (!currentMesh) return alert("Generate first!");

  const exporter = new THREE.OBJExporter();
  const text = exporter.parse(currentMesh);

  const blob = new Blob([text], { type: 'text/plain' });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "model.obj";
  a.click();
};

// ================= RESET =================
resetBtn.onclick = () => location.reload();

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

(async () => {
  await initScene();
  await loadOptions();
})();
