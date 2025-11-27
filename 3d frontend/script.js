/* ============
   Конфигурация / тестовые URL
   ============ */

// Ссылка на ZIP-файл с текстурами. В реальном проекте backend вернет zip_url
const UPLOADED_ZIP_URL = '/mnt/data/f56f11e1-5f53-4509-9a55-5a4c8cbdcc82.zip';

// API endpoints (поставьте реальные URL бекенда позже)
// GET /api/options  -> возвращает { shapes:[], textures:[], colors:[] }
// POST /api/check-or-add -> принимает {shape,color,texture,description} и возвращает {status, zip_url}
const API_GET_OPTIONS = '/api/options';            
const API_CHECK_OR_ADD = '/api/check-or-add';      

/* ============
   DOM элементы
   ============ */
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

/* ============
   Статические опции (fallback если нет API)
   ============ */
const SHAPES = [
  {label:'Cube', value:'Cube'},
  {label:'Sphere', value:'Sphere'},
  {label:'Pyramid', value:'Pyramid'},
  {label:'Prism', value:'Prism'},
  {label:'Cylinder', value:'Cylinder'},
  {label:'Cone', value:'Cone'},
  {label:'Torus', value:'Torus'}
];
const TEXTURES = [
  {label:'Stone', value:'stone'},
  {label:'Metal', value:'metal'},
  {label:'Wood', value:'wood'}
];

/* ============
   Three.js setup
   ============ */
let scene, camera, renderer, currentMesh = null;
function initThree(){
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(55, container.clientWidth/container.clientHeight, 0.1, 1000);
  camera.position.set(0,0,6);

  renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  container.innerHTML = '';
  container.appendChild(renderer.domElement);

  // Свет
  const hemi = new THREE.HemisphereLight(0xffffff, 0x222222, 0.6);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(5,10,7.5);
  scene.add(dir);
}

/* ============
   Геометрии
   ============ */
function createGeometryByShape(shape){
  switch(shape){
    case 'Cube': return new THREE.BoxGeometry(2,2,2);
    case 'Sphere': return new THREE.SphereGeometry(1.2,64,64);
    case 'Pyramid': return new THREE.ConeGeometry(1.2,2.2,4);
    case 'Prism': return new THREE.CylinderGeometry(1.0,1.0,2.0,6);
    case 'Cylinder': return new THREE.CylinderGeometry(1.0,1.0,2.0,32);
    case 'Cone': return new THREE.ConeGeometry(1.0,2.0,32);
    case 'Torus': return new THREE.TorusGeometry(1.0,0.35,30,200);
    default: return new THREE.BoxGeometry(2,2,2);
  }
}

/* ============
   Загрузка текстур из ZIP
   ============ */
async function loadTextureZip(url, desiredTextureKey){
  try{
    const res = await fetch(url);
    if(!res.ok) throw new Error('Failed to load ZIP: ' + res.status);
    const buf = await res.arrayBuffer();
    const zip = await JSZip.loadAsync(buf);

    const files = Object.keys(zip.files);
    const candidates = { diffuse: null, normal: null, roughness: null, metallic: null, ao: null };

    function containsAny(name, arr){ return arr.some(a=>name.includes(a)); }

    // выбираем файлы по ключевому слову
    for(const f of files){
      const name = f.toLowerCase();
      if(name.endsWith('/') || name.includes('__macosx')) continue;

      if(desiredTextureKey && name.includes(desiredTextureKey)){
        if(containsAny(name, ['diffuse','albedo','color','basecolor'])) candidates.diffuse = f;
        if(containsAny(name, ['normal'])) candidates.normal = f;
        if(containsAny(name, ['roughness','rough'])) candidates.roughness = f;
        if(containsAny(name, ['metal','metallic'])) candidates.metallic = f;
        if(containsAny(name, ['ao','ambient'])) candidates.ao = f;
      } else {
        if(!candidates.diffuse && containsAny(name, ['diffuse','albedo','color','basecolor'])) candidates.diffuse = f;
        if(!candidates.normal && containsAny(name, ['normal','normal-ogl','nrm'])) candidates.normal = f;
        if(!candidates.roughness && containsAny(name, ['roughness','rough'])) candidates.roughness = f;
        if(!candidates.metallic && containsAny(name, ['metal','metallic'])) candidates.metallic = f;
        if(!candidates.ao && containsAny(name, ['ao','ambient'])) candidates.ao = f;
      }
    }

    if(desiredTextureKey && !candidates.diffuse){
      for(const f of files){
        const name = f.toLowerCase();
        if(name.includes(desiredTextureKey) && containsAny(name, ['diffuse','albedo','color','basecolor'])) { candidates.diffuse = f; break; }
      }
    }

    const loader = new THREE.TextureLoader();
    async function loadTex(entryName){
      if(!entryName) return null;
      const blob = await zip.file(entryName).async('blob');
      return loader.load(URL.createObjectURL(blob));
    }

    const matParams = {};
    if(candidates.diffuse) matParams.map = await loadTex(candidates.diffuse);
    if(candidates.normal) matParams.normalMap = await loadTex(candidates.normal);
    if(candidates.roughness) matParams.roughnessMap = await loadTex(candidates.roughness);
    if(candidates.metallic) matParams.metalnessMap = await loadTex(candidates.metallic);
    if(candidates.ao) matParams.aoMap = await loadTex(candidates.ao);

    matParams.metalness = (desiredTextureKey === 'metal') ? 1.0 : 0.0;
    matParams.roughness = (desiredTextureKey === 'stone') ? 1.0 : (desiredTextureKey === 'wood' ? 0.8 : 0.4);

    return new THREE.MeshStandardMaterial(matParams);

  } catch(err){
    console.warn('loadTextureZip error', err);
    return null;
  }
}

/* ============
   Применить текстуру к уже существующему объекту
   ============ */
async function applyTextureToExistingMesh(mesh, textureKey, zipUrl){
  if(!mesh) return;

  const mat = await loadTextureZip(zipUrl, textureKey);
  if(!mat){
    console.warn('Texture material not loaded');
    return;
  }

  // сохраняем текущий цвет
  const currentColor = mesh.material?.color || new THREE.Color('#ffffff');
  mat.color = new THREE.Color(currentColor);

  mesh.traverse(child=>{
    if(child.isMesh){
      if(mat.aoMap && !child.geometry.attributes.uv2){
        child.geometry.setAttribute(
          'uv2',
          new THREE.BufferAttribute(child.geometry.attributes.uv.array, 2)
        );
      }
      child.material = mat;
    }
  });
}

/* ============
   Создание/обновление меша
   ============ */
async function createMesh(shape, colorHex, textureKey, zipUrl){
  const geo = createGeometryByShape(shape);

  const baseMat = new THREE.MeshStandardMaterial({
    color: colorHex,
    roughness: (textureKey==='stone')?1.0:0.6,
    metalness: (textureKey==='metal')?0.9:0.0
  });

  const mesh = new THREE.Mesh(geo, baseMat);

  if(zipUrl){
    const mat = await loadTextureZip(zipUrl, textureKey);
    if(mat){
      mat.color = new THREE.Color(colorHex);
      if(mat.aoMap && !geo.attributes.uv2){
        geo.setAttribute('uv2', new THREE.BufferAttribute(geo.attributes.uv.array, 2));
      }
      mesh.material = mat;
    }
  }
  return mesh;
}

/* ============
   Обновление объекта
   ============ */
async function updateObject(shape, colorHex, textureKey){
  if(!currentMesh){
    currentMesh = await createMesh(shape, colorHex, textureKey, UPLOADED_ZIP_URL);
    scene.add(currentMesh);
    return;
  }

  const currentShapeName = currentMesh.userData.shapeName || null;
  if(currentShapeName !== shape){
    scene.remove(currentMesh);
    currentMesh.geometry && currentMesh.geometry.dispose && currentMesh.geometry.dispose();
    currentMesh = await createMesh(shape, colorHex, textureKey, UPLOADED_ZIP_URL);
    currentMesh.userData.shapeName = shape;
    scene.add(currentMesh);
    return;
  }

  // обновляем цвет
  currentMesh.traverse(child=>{
    if(child.isMesh && child.material && child.material.color) {
      child.material.color.set(colorHex);
    }
  });

  // обновляем текстуру
  await applyTextureToExistingMesh(currentMesh, textureKey, UPLOADED_ZIP_URL);
}

/* ============
   animation loop (две оси)
   ============ */
function animate(){
  requestAnimationFrame(animate);

  if(currentMesh){
    currentMesh.rotation.x += 0.01; // вертикальное вращение
    currentMesh.rotation.y += 0.01; // горизонтальное вращение
  }

  renderer.render(scene, camera);
}

/* ============
   Dropdown UI
   ============ */
function setupDropdown(dropdown, optionsDiv, optionsArr, onSelect){
  optionsDiv.innerHTML = '';
  optionsArr.forEach(opt=>{
    const el = document.createElement('div');
    el.textContent = opt.label;
    el.dataset.value = opt.value;
    el.addEventListener('click', ()=>{
      Array.from(optionsDiv.children).forEach(c => { c.style.background=''; c.style.color='#fff'; });
      el.style.background = 'linear-gradient(90deg,#4b00e0,#8e2de2)';
      el.style.color = '#fff';
      dropdown.classList.remove('show');
      onSelect(opt.value, opt.label);
    });
    optionsDiv.appendChild(el);
  });

  const btn = dropdown.querySelector('.dropbtn');
  btn.addEventListener('click', e=>{
    e.stopPropagation();
    document.querySelectorAll('.dropdown').forEach(d=>{
      if(d!==dropdown) d.classList.remove('show');
    });
    dropdown.classList.toggle('show');
  });
}

window.addEventListener('click', ()=> 
  document.querySelectorAll('.dropdown').forEach(d=>d.classList.remove('show'))
);

/* ============
   AI parser (только английский)
   ============ */
function aiGenerate(description){
  const txt = (description||'').toLowerCase();
  const shapeMap = {
    "cube":"Cube",
    "sphere":"Sphere",
    "ball":"Sphere",
    "pyramid":"Pyramid",
    "prism":"Prism",
    "cylinder":"Cylinder",
    "cone":"Cone",
    "torus":"Torus"
  };
  const texMap = {
    "metal":"metal",
    "wood":"wood",
    "stone":"stone"
  };
  const colorMap = {
    "red":"#FF4B4B",
    "blue":"#4B69FF",
    "green":"#00C976",
    "yellow":"#FFD700",
    "purple":"#8A70D6",
    "orange":"#FF8C42",
    "pink":"#FF69B4",
    "turquoise":"#00CED1",
    "white":"#FFFFFF",
    "black":"#000000"
  };

  let shape='Cube', texture='stone', color='#6952BE';
  for(const k in shapeMap) if(txt.includes(k)) shape = shapeMap[k];
  for(const k in texMap) if(txt.includes(k)) texture = texMap[k];
  for(const k in colorMap) if(txt.includes(k)) color = colorMap[k];

  return { shape, texture, color };
}

/* ============
   Backend communication
   - POST /api/check-or-add
   - Backend должен:
     1. Проверить уникальность по shape+color+texture
     2. Если новая модель -> создать zip (OBJ+MTL+texture) и вернуть zip_url
     3. Если есть -> вернуть zip_url существующего
   ============ */
async function sendParamsToBackend(params){
  try{
    const res = await fetch(API_CHECK_OR_ADD, {
      method:'POST',
      headers:{ 'Content-Type':'application/json' },
      body: JSON.stringify(params)
    });
    if(!res.ok) return null;
    const data = await res.json();
    return data;
  } catch(err){
    console.warn('sendParamsToBackend error:', err);
    return null;
  }
}

/* ============
   UI wiring
   ============ */
let selectedShape = 'Cube';
let selectedTexture = 'stone';

function loadUI(){
  colorInput.value = '#6952BE';

  (async ()=>{
    try{
      const res = await fetch(API_GET_OPTIONS);
      if(res.ok){
        const data = await res.json();
        const shapes = (data.shapes && data.shapes.length) ? data.shapes.map(s=>({label:s,value:s})) : SHAPES;
        const textures = (data.textures && data.textures.length) ? data.textures.map(t=>({label:t.charAt(0).toUpperCase()+t.slice(1),value:t})) : TEXTURES;
        setupDropdown(shapeDropdown, shapeOptionsDiv, shapes, (val,label)=>{
          selectedShape = val;
          shapeDropdown.querySelector('.dropbtn').textContent = label;
          updateObject(selectedShape, colorInput.value, selectedTexture);
        });
        setupDropdown(textureDropdown, textureOptionsDiv, textures, (val,label)=>{
          selectedTexture = val;
          textureDropdown.querySelector('.dropbtn').textContent = label;
          applyTextureToExistingMesh(currentMesh, selectedTexture, UPLOADED_ZIP_URL);
        });
        if(data.colors && data.colors.length) colorInput.value = data.colors[0];
        return;
      }
    }catch(e){}
    setupDropdown(shapeDropdown, shapeOptionsDiv, SHAPES, (val,label)=>{
      selectedShape = val;
      shapeDropdown.querySelector('.dropbtn').textContent = label;
      updateObject(selectedShape, colorInput.value, selectedTexture);
    });
    setupDropdown(textureDropdown, textureOptionsDiv, TEXTURES, (val,label)=>{
      selectedTexture = val;
      textureDropdown.querySelector('.dropbtn').textContent = label;
      applyTextureToExistingMesh(currentMesh, selectedTexture, UPLOADED_ZIP_URL);
    });
  })();
}

// generate button
generateButton.addEventListener('click', async ()=>{
  const { shape, texture, color } = aiGenerate(descriptionInput.value);
  selectedShape = shape;
  selectedTexture = texture;
  colorInput.value = color;

  shapeOptionsDiv.querySelectorAll('div').forEach(d=>{ d.style.background=''; d.style.color='#fff'; if(d.dataset.value===shape) { d.style.background='linear-gradient(90deg,#4b00e0,#8e2de2)'; }});
  textureOptionsDiv.querySelectorAll('div').forEach(d=>{ d.style.background=''; d.style.color='#fff'; if(d.dataset.value===texture) { d.style.background='linear-gradient(90deg,#4b00e0,#8e2de2)'; }});
  shapeDropdown.querySelector('.dropbtn').textContent = shape;
  textureDropdown.querySelector('.dropbtn').textContent = texture;

  await updateObject(shape, color, texture);

  const params = { shape, color, texture, description: descriptionInput.value };
  const data = await sendParamsToBackend(params);
  if(data && data.zip_url){
    try{
      await fetchAndApplyZip(data.zip_url);
    }catch(e){
      console.warn('Failed to fetch zip from backend:', e);
    }
  }
});

// export button
exportButton.addEventListener('click', exportCurrentObjectAsZip);

// reset button
resetButton.addEventListener('click', ()=>{
  descriptionInput.value = '';
  selectedShape = '';
  selectedTexture = '';
  shapeDropdown.querySelector('.dropbtn').textContent = 'Select Shape';
  textureDropdown.querySelector('.dropbtn').textContent = 'Select Texture';
  if(currentMesh){ scene.remove(currentMesh); currentMesh = null; }
});

// color input changes tint
colorInput.addEventListener('input', ()=>{
  if(currentMesh){
    currentMesh.traverse(child=>{
      if(child.isMesh && child.material && child.material.color) child.material.color.set(colorInput.value);
    });
  }
});

// helper: fetch zip from backend
async function fetchAndApplyZip(zipUrl){
  try{
    const res = await fetch(zipUrl);
    if(!res.ok) throw new Error('zip fetch failed ' + res.status);
    const buf = await res.arrayBuffer();
    const zip = await JSZip.loadAsync(buf);

    let objEntry = null, mtlEntry = null, textureEntry = null;
    for(const fname of Object.keys(zip.files)){
      const ln = fname.toLowerCase();
      if(!objEntry && ln.endsWith('.obj')) objEntry = fname;
      if(!mtlEntry && ln.endsWith('.mtl')) mtlEntry = fname;
      if(!textureEntry && (ln.endsWith('.png')||ln.endsWith('.jpg')||ln.endsWith('.jpeg'))) textureEntry = fname;
    }
    if(objEntry){
      const objText = await zip.file(objEntry).async('text');
      let texUrl = null;
      if(textureEntry){
        const blob = await zip.file(textureEntry).async('blob');
        texUrl = URL.createObjectURL(blob);
      }

      if(window.THREE && THREE.OBJLoader){
        const loader = new THREE.OBJLoader();
        const obj = loader.parse(objText);
        if(texUrl){
          const tmap = new THREE.TextureLoader().load(texUrl);
          obj.traverse(c=>{
            if(c.isMesh) c.material = new THREE.MeshStandardMaterial({map: tmap});
          });
        }
        if(currentMesh) scene.remove(currentMesh);
        currentMesh = obj;
        scene.add(currentMesh);
      } else {
        console.warn('OBJLoader not found; using placeholder cube.');
        if(currentMesh) scene.remove(currentMesh);
        currentMesh = new THREE.Mesh(new THREE.BoxGeometry(2,2,2), new THREE.MeshStandardMaterial({color: colorInput.value}));
        scene.add(currentMesh);
      }
    } else {
      console.warn('No OBJ found inside provided zip.');
    }
  } catch(err){
    console.warn('fetchAndApplyZip error', err);
  }
}

/* ============
   init
   ============ */
initThree();
loadUI();
updateObject('Cube', '#6952BE', 'stone').then(()=>{ animate(); });

