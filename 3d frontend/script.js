/* script.js — production-ready frontend for your FastAPI server
   Supports: /api/options, /api/generate-object, /api/generate-from-text
   Handles ZIP containing OBJ + MTL + textures (png/jpg/jpeg)
   Replaces texture references in MTL with blob: URLs and uses MTLLoader
   Uses OBJLoader to parse object, applies materials
   OrbitControls loaded dynamically
*/

/* ======== Конфигурация ======== */
const SERVER_URL = 'http://127.0.0.1:8000';
const API_GET_OPTIONS = SERVER_URL + '/api/options';
const API_GENERATE_OBJECT = SERVER_URL + '/api/generate-object';
const API_GENERATE_TEXT = SERVER_URL + '/api/generate-from-text';
const API_LOG_SHAPE = SERVER_URL + '/api/log-shape';
const API_LOG_TEXTURE = SERVER_URL + '/api/log-texture';
const API_LOG_COLOR = SERVER_URL + '/api/log-color';

/* ======== DOM элементы ======== */
const shapeDropdownRoot = document.getElementById('shapeDropdown');
const shapeBtn = shapeDropdownRoot.querySelector('.dropbtn');
const shapeOptionsContainer = document.getElementById('shapeOptions');

const textureDropdownRoot = document.getElementById('textureDropdown');
const textureBtn = textureDropdownRoot.querySelector('.dropbtn');
const textureOptionsContainer = document.getElementById('textureOptions');

const colorInput = document.getElementById('color');
const descriptionInput = document.getElementById('description');
const generateButton = document.getElementById('generate');
const exportButton = document.getElementById('export');
const resetButton = document.getElementById('reset');
const container = document.getElementById('objectPreview');

/* ======== Three.js vars ======== */
let scene, camera, renderer, orbitControls, currentMesh = null;
let tempBlobUrls = [];

/* ======== Helper: load OrbitControls dynamically ======== */
function loadOrbitControls(callback){
  if (window.THREE && THREE.OrbitControls) { callback(); return; }
  const s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.js';
  s.onload = () => callback();
  s.onerror = () => { console.warn('OrbitControls failed to load'); callback(); };
  document.head.appendChild(s);
}

/* ======== Initialization of Three.js scene ======== */
function initThree(){
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.set(0, 0, 6);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.innerHTML = '';
  container.appendChild(renderer.domElement);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x222222, 0.6);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(5, 10, 7.5);
  scene.add(dir);

  loadOrbitControls(() => {
    if (THREE.OrbitControls) {
      orbitControls = new THREE.OrbitControls(camera, renderer.domElement);
      orbitControls.enableDamping = true;
      orbitControls.dampingFactor = 0.07;
      orbitControls.minDistance = 2;
      orbitControls.maxDistance = 30;
    }
  });

  window.addEventListener('resize', onWindowResize);
}

function onWindowResize(){
  if (!camera || !renderer) return;
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

/* ======== Dropdown UI utilities ======== */
function createOptionElement(label, value, onClick){
  const el = document.createElement('div');
  el.textContent = label;
  el.dataset.value = value;
  el.addEventListener('click', () => {
    onClick(value, label);
    el.closest('.dropdown').classList.remove('show');
  });
  return el;
}

function toggleDropdown(root){ root.classList.toggle('show'); }
document.addEventListener('click', (e) => {
  for (const dd of document.querySelectorAll('.dropdown')) {
    if (!dd.contains(e.target)) dd.classList.remove('show');
  }
});

shapeBtn.addEventListener('click', () => toggleDropdown(shapeDropdownRoot));
textureBtn.addEventListener('click', () => toggleDropdown(textureDropdownRoot));

/* ======== State ======== */
let selectedShape = null;
let selectedTexture = null;

/* ======== Load options from server and populate dropdowns ======== */
async function loadOptionsFromServer(){
  try {
    const res = await fetch(API_GET_OPTIONS);
    if (!res.ok) throw new Error('Failed to fetch options: ' + res.status);
    const data = await res.json();

    const shapes = Array.isArray(data.shapes) ? data.shapes : [];
    const textures = Array.isArray(data.textures) ? data.textures : [];

    populateShapeOptions(shapes);
    populateTextureOptions(textures);

    if (shapes.length) selectShape(shapes[0]);
    if (textures.length) selectTexture(textures[0]);

  } catch (err) {
    console.warn('loadOptionsFromServer error:', err);
    const fallbackShapes = ["cube","sphere" ,"pyramid","Prism","cylinder","cone","torus"];
    const fallbackTextures = ["stone","metal","wood", "none"];
    populateShapeOptions(fallbackShapes);
    populateTextureOptions(fallbackTextures);
    selectShape(fallbackShapes[0]);
    selectTexture(fallbackTextures[0]);
  }
}

function populateShapeOptions(list){
  shapeOptionsContainer.innerHTML = '';
  list.forEach(item => {
    const label = typeof item === 'string' ? item : item.label || item.value;
    const value = typeof item === 'string' ? item : item.value || item.label;
    shapeOptionsContainer.appendChild(createOptionElement(label, value, selectShape));
  });
}

function populateTextureOptions(list){
  textureOptionsContainer.innerHTML = '';
  list.forEach(item => {
    const label = typeof item === 'string' ? item : item.label || item.value;
    const value = typeof item === 'string' ? item : item.value || item.label;
    textureOptionsContainer.appendChild(createOptionElement(label, value, selectTexture));
  });
}

function selectShape(value, label){
  selectedShape = value;
  shapeBtn.textContent = label || value;
  fetch(API_LOG_SHAPE, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({shape: value}) }).catch(()=>{});
  if (!currentMesh || (currentMesh.userData && !currentMesh.userData.generatedFromServer)) {
    updatePreviewGeometry(selectedShape, colorInput.value);
  }
}

function selectTexture(value, label){
  selectedTexture = value;
  textureBtn.textContent = label || value;
  fetch(API_LOG_TEXTURE, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({texture: value}) }).catch(()=>{});
  if (!currentMesh || (currentMesh.userData && !currentMesh.userData.generatedFromServer)) {
    applyTextureToCurrentMesh(null);
  }
}

/* ======== Local geometry preview ======== */
function createGeometryByShape(shape){
  switch (shape) {
    case 'cube': return new THREE.BoxGeometry(2,2,2);
    case 'sphere': return new THREE.SphereGeometry(1.2,64,64);
    case 'pyramid': return new THREE.ConeGeometry(1.2,2.2,4);
    case 'prism': return new THREE.CylinderGeometry(1,1,2,6);
    case 'cylinder': return new THREE.CylinderGeometry(1,1,2,32);
    case 'cone': return new THREE.ConeGeometry(1,2,32);
    case 'torus': return new THREE.TorusGeometry(1,0.35,30,200);
    default: return new THREE.BoxGeometry(2,2,2);
  }
}

function updatePreviewGeometry(shape, colorHex){
  if (currentMesh && currentMesh.userData && currentMesh.userData.generatedFromServer) return;
  if (currentMesh) {
    scene.remove(currentMesh);
    disposeObject(currentMesh);
    currentMesh = null;
  }
  const geo = createGeometryByShape(shape);
  const mat = new THREE.MeshStandardMaterial({ color: colorHex || '#6952BE' });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.userData = { generatedFromServer: false };
  currentMesh = mesh;
  scene.add(currentMesh);
}

/* ======== Apply texture/color to local mesh ======== */
function applyTextureToCurrentMesh(textureUrl){
  if (!currentMesh) return;
  if (currentMesh.userData && currentMesh.userData.generatedFromServer) return;
  currentMesh.traverse(n => {
    if (n.isMesh) {
      if (textureUrl) {
        const tloader = new THREE.TextureLoader();
        tloader.load(textureUrl, tex => { n.material = new THREE.MeshStandardMaterial({ map: tex }); });
      } else {
        n.material = new THREE.MeshStandardMaterial({ color: colorInput.value || '#6952BE' });
      }
    }
  });
}

/* ======== Dispose mesh ======== */
function disposeObject(obj){
  obj.traverse(o => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      if (Array.isArray(o.material)) o.material.forEach(m => m.dispose());
      else o.material.dispose();
    }
  });
}

/* ======== Fetch ZIP and apply OBJ/MTL ======== */
async function fetchAndApplyZip(zipUrl){
  try {
    if (zipUrl.startsWith('/')) zipUrl = SERVER_URL + zipUrl;
    tempBlobUrls.forEach(u => URL.revokeObjectURL(u));
    tempBlobUrls = [];

    const res = await fetch(zipUrl);
    if (!res.ok) throw new Error('ZIP fetch failed: ' + res.status);
    const buf = await res.arrayBuffer();
    const zip = await JSZip.loadAsync(buf);

    let objEntry = null, mtlEntry = null;
    const textureMap = {};

    for (const fname of Object.keys(zip.files)) {
      const lower = fname.toLowerCase();
      if (!objEntry && lower.endsWith('.obj')) objEntry = fname;
      if (!mtlEntry && lower.endsWith('.mtl')) mtlEntry = fname;
      if (lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg')) {
        const blob = await zip.file(fname).async('blob');
        const url = URL.createObjectURL(blob);
        const base = fname.split('/').pop();
        textureMap[base] = url;
        textureMap[fname] = url;
        tempBlobUrls.push(url);
      }
    }

    if (!objEntry) { console.warn('ZIP contains no .obj'); return; }

    const objText = await zip.file(objEntry).async('text');
    let materialsCreator = null;

    if (mtlEntry) {
      let mtlText = await zip.file(mtlEntry).async('text');
      for (const key of Object.keys(textureMap)) {
        const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        mtlText = mtlText.replace(new RegExp(escaped, 'g'), textureMap[key]);
      }
      try {
        const mtlLoader = new THREE.MTLLoader();
        materialsCreator = mtlLoader.parse(mtlText, '');
        materialsCreator.preload();
      } catch(e) { console.warn('MTL parse failed:', e); materialsCreator=null; }
    }

    const objLoader = new THREE.OBJLoader();
    if (materialsCreator) objLoader.setMaterials(materialsCreator);

    let obj;
    try { obj = objLoader.parse(objText); }
    catch(e) {
      const blob = new Blob([objText], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      tempBlobUrls.push(url);
      obj = await new Promise((resolve, reject) => {
        objLoader.load(url, o=>{URL.revokeObjectURL(url); resolve(o)}, undefined, reject);
      });
    }

    if (!materialsCreator && Object.keys(textureMap).length) {
      const texUrl = Object.values(textureMap)[0];
      const tloader = new THREE.TextureLoader();
      const texture = await new Promise((res, rej)=>tloader.load(texUrl, res, undefined, rej));
      obj.traverse(c=>{if(c.isMesh) c.material=new THREE.MeshStandardMaterial({map: texture});});
    }

    if (currentMesh) { scene.remove(currentMesh); disposeObject(currentMesh); currentMesh=null; }

    obj.userData = { generatedFromServer:true };
    currentMesh = obj;
    scene.add(currentMesh);
    centerAndScaleObject(currentMesh);

  } catch(err) {
    console.error('fetchAndApplyZip error:', err);
    alert('Ошибка при загрузке модели: ' + (err.message||err));
  }
}

/* ======== Center and scale object ======== */
function centerAndScaleObject(obj){
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x,size.y,size.z);
  if(maxDim>0) obj.scale.setScalar(2.2/maxDim);
  const center = box.getCenter(new THREE.Vector3());
  obj.position.sub(center.multiplyScalar(obj.scale.x));
}

/* ======== Generate object ======== */
generateButton.addEventListener('click', async()=>{
  const description = descriptionInput.value.trim();
  let endpoint, payload;

  if(description){
    endpoint = API_GENERATE_TEXT;
    payload = { text: description };
  } else {
    endpoint = API_GENERATE_OBJECT;
    const shape = selectedShape||(shapeOptionsContainer.firstChild && shapeOptionsContainer.firstChild.dataset.value)||'Cube';
    const texture = selectedTexture||(textureOptionsContainer.firstChild && textureOptionsContainer.firstChild.dataset.value)||'stone';
    const color = colorInput.value||'#6952BE';
    payload = { shape, texture, color };
  }

  generateButton.disabled = true;
  generateButton.textContent = 'Generating...';

  try{
    const res = await fetch(endpoint,{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    if(!res.ok){
      const text = await res.text().catch(()=>null);
      throw new Error('Server error '+res.status+(text?': '+text:''));
    }
    const data = await res.json();
    if(data.zip_url){ await fetchAndApplyZip(data.zip_url); }
    else if(data.obj_url){ await loadObjDirect(data.obj_url); }
    else { alert('Не получили ссылку на модель от сервера.'); }

    if(endpoint===API_GENERATE_OBJECT){
      fetch(API_LOG_COLOR,{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({color:payload.color})}).catch(()=>{});
    }
  } catch(err){ console.error('Generation failed:',err); alert('Ошибка генерации: '+(err.message||err)); }
  finally{ generateButton.disabled=false; generateButton.textContent='Generate'; }
});

/* ======== Load OBJ directly ======== */
async function loadObjDirect(objUrl){
  try{
    const objLoader = new THREE.OBJLoader();
    const obj = await new Promise((resolve,reject)=>objLoader.load(objUrl,resolve,undefined,reject));
    if(currentMesh){ scene.remove(currentMesh); disposeObject(currentMesh); currentMesh=null; }
    obj.userData={generatedFromServer:true};
    currentMesh=obj;
    scene.add(currentMesh);
    centerAndScaleObject(currentMesh);
  } catch(err){ console.error('loadObjDirect error:',err); alert('Ошибка при загрузке OBJ: '+(err.message||err)); }
}

/* ======== Export mesh to ZIP ======== */
exportButton.addEventListener('click', async()=>{
  if(!currentMesh){ alert('Сначала сгенерируйте объект'); return; }
  try{
    const exporter = new THREE.OBJExporter();
    const objText = exporter.parse(currentMesh);
    const mtlText = 'newmtl material_0\nKd 1 1 1\n';
    const zip = new JSZip();
    zip.file('model.obj', objText);
    zip.file('material.mtl', mtlText);
    const blob = await zip.generateAsync({type:'blob'});
    const url = URL.createObjectURL(blob);
    const a=document.createElement('a'); a.href=url; a.download='model_export.zip'; document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch(err){ console.error('Export failed:',err); alert('Ошибка экспорта: '+(err.message||err)); }
});

/* ======== Reset UI ======== */
resetButton.addEventListener('click', ()=>{
  descriptionInput.value='';
  colorInput.value='#6952BE';
  if(shapeOptionsContainer.firstChild){ selectShape(shapeOptionsContainer.firstChild.dataset.value, shapeOptionsContainer.firstChild.textContent); }
  if(textureOptionsContainer.firstChild){ selectTexture(textureOptionsContainer.firstChild.dataset.value, textureOptionsContainer.firstChild.textContent); }
  if(currentMesh){ scene.remove(currentMesh); disposeObject(currentMesh); currentMesh=null; }
});

/* ======== Color change ======== */
colorInput.addEventListener('input', ()=>{
  fetch(API_LOG_COLOR,{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({color:colorInput.value})}).catch(()=>{});
  if(!currentMesh) return;
  if(currentMesh.userData && currentMesh.userData.generatedFromServer) return;
  applyTextureToCurrentMesh(null);
});

/* ======== Animation loop ======== */
function animate(){ requestAnimationFrame(animate); if(orbitControls) orbitControls.update(); if(currentMesh && !(currentMesh.userData && currentMesh.userData.generatedFromServer)){ currentMesh.rotation.x+=0.008; currentMesh.rotation.y+=0.01; } renderer.render(scene,camera); }

/* ======== Boot ======== */
(function boot(){
  initThree();
  loadOptionsFromServer();
  animate();
  descriptionInput.addEventListener('keydown',(e)=>{ if(e.key==='Enter' && (e.ctrlKey||e.metaKey)) generateButton.click(); });
})();
