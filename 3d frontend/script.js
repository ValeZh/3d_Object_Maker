/* ============ Настройка сервера ============ */
const SERVER_URL = 'http://127.0.0.1:8000';
const API_GET_OPTIONS = SERVER_URL + '/api/options';
const API_GENERATE_OBJECT = SERVER_URL + '/api/generate-object';

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

    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
}

/* ============ Load / Update mesh с fade-in ============ */
async function createMeshFromBackend(shape, colorHex, texture, description = null) {
    try {
        console.log('Запрос к серверу для генерации объекта...');
        const payload = {};
        if (description && description.trim().length > 0) {
            payload.description = description.trim();
        } else {
            payload.shape = shape;
            payload.color = colorHex;
            payload.texture = texture;
        }

        const res = await fetch(API_GENERATE_OBJECT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const data = await res.json();

        const objUrl = SERVER_URL + data.obj_url.replace(/\\/g, '/');
        const mtlUrl = SERVER_URL + data.mtl_url.replace(/\\/g, '/');

        const mtlLoader = new THREE.MTLLoader();
        const mtl = await new Promise((resolve, reject) => {
            mtlLoader.load(mtlUrl, resolve, undefined, reject);
        });
        mtl.preload();

        const objLoader = new THREE.OBJLoader();
        objLoader.setMaterials(mtl);
        const obj = await new Promise((resolve, reject) => {
            objLoader.load(objUrl, resolve, undefined, reject);
        });

        obj.position.set(0, 0, 0);
        const box = new THREE.Box3().setFromObject(obj);
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        if (maxDim > 0) obj.scale.setScalar(2 / maxDim);

        // Устанавливаем прозрачность для плавного появления
        obj.traverse(child => {
            if (child.isMesh) {
                child.material.transparent = true;
                child.material.opacity = 0;
            }
        });

        console.log('Объект загружен и готов к отображению.');
        return obj;
    } catch (err) {
        console.error('Ошибка создания mesh с сервера:', err);
        alert('Ошибка при загрузке 3D объекта. Попробуйте еще раз.');
        return null;
    }
}

async function updateObject(shape, colorHex, texture, description = null) {
    if (currentMesh) {
        currentMesh.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) child.material.forEach(mat => mat.dispose());
                else child.material.dispose();
            }
        });
        scene.remove(currentMesh);
        currentMesh = null;
    }

    const newMesh = await createMeshFromBackend(shape, colorHex, texture, description);
    if (newMesh) {
        scene.add(newMesh);
        currentMesh = newMesh;

        // Анимация плавного появления
        const duration = 30; // кадров
        let frame = 0;
        function fadeIn() {
            frame++;
            newMesh.traverse(child => {
                if (child.isMesh) child.material.opacity = Math.min(frame / duration, 1);
            });
            if (frame < duration) requestAnimationFrame(fadeIn);
        }
        fadeIn();
    }
}

/* ============ Animation loop ============ */
function animate() {
    requestAnimationFrame(animate);
    if (currentMesh) {
        currentMesh.rotation.x += 0.005; // мягкое вращение
        currentMesh.rotation.y += 0.007;
    }
    renderer.render(scene, camera);
}

/* ============ Dropdown helper ============ */
function setupDropdown(dropdown, optionsDiv, optionsArr, onSelect, defaultValue = null) {
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

    if (defaultValue) {
        const selectedEl = Array.from(optionsDiv.children).find(c => c.dataset.value === defaultValue);
        if (selectedEl) {
            selectedEl.style.background = 'linear-gradient(90deg,#4b00e0,#8e2de2)';
            selectedEl.style.color = '#fff';
            btn.textContent = selectedEl.textContent;
        }
    }
}

window.addEventListener('click', () => document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('show')));

/* ============ Load UI from backend ============ */
let selectedShape = 'Cube';
let selectedTexture = 'stone';

async function loadUI() {
    colorInput.value = '#6952BE';
    try {
        const res = await fetch(API_GET_OPTIONS);
        if (!res.ok) throw new Error('Cannot load options');
        const data = await res.json();

        const shapes = data.shapes.map(s => ({ label: s, value: s }));
        const textures = data.textures.map(t => ({ label: t, value: t }));

        setupDropdown(shapeDropdown, shapeOptionsDiv, shapes, (val, label) => {
            selectedShape = val;
            shapeDropdown.querySelector('.dropbtn').textContent = label;
            updateObject(selectedShape, colorInput.value, selectedTexture);
        }, selectedShape);

        setupDropdown(textureDropdown, textureOptionsDiv, textures, (val, label) => {
            selectedTexture = val;
            textureDropdown.querySelector('.dropbtn').textContent = label;
            updateObject(selectedShape, colorInput.value, selectedTexture);
        }, selectedTexture);
    } catch (err) {
        console.warn(err);
        alert('Ошибка при загрузке опций с сервера.');
    }
}

/* ============ Buttons с логами ============ */
generateButton.addEventListener('click', async () => {
    console.log('Нажата кнопка генерации.');
    const desc = descriptionInput.value.trim();
    if (desc.length > 0) {
        generateButton.disabled = true;
        generateButton.textContent = 'Generating...';
        try {
            await updateObject(null, colorInput.value, null, desc);
        } catch (err) {
            console.error(err);
            alert('Ошибка генерации объекта по AI-описанию.');
        } finally {
            generateButton.disabled = false;
            generateButton.textContent = 'Generate';
        }
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
    if (currentMesh) {
        currentMesh.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        });
        scene.remove(currentMesh);
        currentMesh = null;
    }
});

colorInput.addEventListener('input', () => {
    if (currentMesh) {
        currentMesh.traverse(child => { 
            if (child.isMesh && child.material && child.material.color) child.material.color.set(colorInput.value); 
        });
    }
});

exportButton.addEventListener('click', () => {
    if (!currentMesh) return alert('No object to export!');

    const exporter = new THREE.OBJExporter();
    const objData = exporter.parse(currentMesh);

    const blob = new Blob([objData], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedShape || 'object'}.obj`;
    a.click();
    URL.revokeObjectURL(url);
});

/* ============ Init ============ */
initThree();
loadUI();
updateObject('Cube', '#6952BE', 'stone').then(() => animate());
