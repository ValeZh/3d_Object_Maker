const SERVER_URL = location.origin;

// ================= UI TABS =================
const tabModules = document.getElementById("tabModules");
const tabLibrary = document.getElementById("tabLibrary");
const tabBuilder = document.getElementById("tabBuilder");

const modulePage = document.getElementById("modulePage");
const libraryPage = document.getElementById("libraryPage");
const builderPage = document.getElementById("builderPage");

const resetBtn = document.getElementById("reset");

// ================= MODULE GENERATOR UI =================
const moduleDescription = document.getElementById("moduleDescription");
const analyzeModuleBtn = document.getElementById("analyzeModule");
const moduleName = document.getElementById("moduleName");
const moduleType = document.getElementById("moduleType");
const moduleColor = document.getElementById("moduleColor");
const moduleWidth = document.getElementById("moduleWidth");
const moduleHeight = document.getElementById("moduleHeight");
const moduleDepth = document.getElementById("moduleDepth");
const moduleHasWindow = document.getElementById("moduleHasWindow");
const moduleHasDoor = document.getElementById("moduleHasDoor");
const moduleHasBalcony = document.getElementById("moduleHasBalcony");
const moduleIsRoofPiece = document.getElementById("moduleIsRoofPiece");
const generateModuleBtn = document.getElementById("generateModule");
const saveModuleBtn = document.getElementById("saveModule");
const moduleJson = document.getElementById("moduleJson");

// ================= MODULE LIBRARY UI =================
const libraryFilter = document.getElementById("libraryFilter");
const savedModules = document.getElementById("savedModules");

// ================= HOUSE BUILDER UI =================
const houseDescription = document.getElementById("houseDescription");
const analyzeHouseBtn = document.getElementById("analyzeHouse");

const floorsInput = document.getElementById("floors");
const sectionsInput = document.getElementById("sections");
const widthInput = document.getElementById("width");
const depthInput = document.getElementById("depth");
const houseColor = document.getElementById("houseColor");
const roofType = document.getElementById("roofType");
const hasBalconies = document.getElementById("hasBalconies");
const balconyRate = document.getElementById("balconyRate");
const windowCols = document.getElementById("windowCols");

const floorsValue = document.getElementById("floorsValue");
const sectionsValue = document.getElementById("sectionsValue");
const widthValue = document.getElementById("widthValue");
const depthValue = document.getElementById("depthValue");
const balconyRateValue = document.getElementById("balconyRateValue");
const windowColsValue = document.getElementById("windowColsValue");

const wallModule = document.getElementById("wallModule");
const windowModule = document.getElementById("windowModule");
const doorModule = document.getElementById("doorModule");
const roofModule = document.getElementById("roofModule");
const balconyModule = document.getElementById("balconyModule");

const generateHouseBtn = document.getElementById("generateHouse");
const animateHouseBtn = document.getElementById("animateHouse");
const compareHouseBtn = document.getElementById("compareHouse");
const exportZipBtn = document.getElementById("exportZip");
const houseJson = document.getElementById("houseJson");

// ================= PREVIEW =================
const preview = document.getElementById("objectPreview");
const previewJson = document.getElementById("previewJson");
const hoverInfo = document.getElementById("hoverInfo");

// ================= STATE =================
let scene, camera, renderer, controls;
let currentMesh = null;
let compareMesh = null;
let savedModulesData = [];
let latestExportZipBlob = null;
let interactiveObjects = [];
let raycaster, mouse;

// ================= THREE =================
async function waitForThree() {
  return new Promise(resolve => {
    const check = () => {
      if (window.THREE && window.THREE.OrbitControls) resolve();
      else setTimeout(check, 50);
    };
    check();
  });
}

async function initScene() {
  await waitForThree();

  const THREE = window.THREE;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x121212);

  camera = new THREE.PerspectiveCamera(
    60,
    preview.clientWidth / preview.clientHeight,
    0.1,
    1000
  );
  camera.position.set(18, 12, 18);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(preview.clientWidth, preview.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  preview.appendChild(renderer.domElement);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x333333, 1.5);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 1.1);
  dir.position.set(10, 20, 10);
  scene.add(dir);

  const grid = new THREE.GridHelper(100, 100, 0x333333, 0x222222);
  scene.add(grid);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 4, 0);

  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  preview.addEventListener("mousemove", onPointerMove);
  preview.addEventListener("mouseleave", () => hoverInfo.classList.add("hidden"));

  window.addEventListener("resize", () => {
    camera.aspect = preview.clientWidth / preview.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(preview.clientWidth, preview.clientHeight);
  });

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
}

// ================= HELPERS =================
function clearMesh(mesh) {
  if (!mesh) return;

  scene.remove(mesh);
  mesh.traverse(obj => {
    if (obj.geometry) obj.geometry.dispose?.();
    if (obj.material) {
      if (Array.isArray(obj.material)) {
        obj.material.forEach(m => m.dispose?.());
      } else {
        obj.material.dispose?.();
      }
    }
  });
}

function clearSceneMeshes() {
  if (currentMesh) clearMesh(currentMesh);
  if (compareMesh) clearMesh(compareMesh);
  currentMesh = null;
  compareMesh = null;
  interactiveObjects = [];
}

function setPreviewJson(data) {
  previewJson.textContent = JSON.stringify(data, null, 2);
}

function setActiveTab(tab) {
  modulePage.classList.add("hidden");
  libraryPage.classList.add("hidden");
  builderPage.classList.add("hidden");

  tabModules.classList.remove("active-tab");
  tabLibrary.classList.remove("active-tab");
  tabBuilder.classList.remove("active-tab");

  if (tab === "modules") {
    modulePage.classList.remove("hidden");
    tabModules.classList.add("active-tab");
  } else if (tab === "library") {
    libraryPage.classList.remove("hidden");
    tabLibrary.classList.add("active-tab");
  } else {
    builderPage.classList.remove("hidden");
    tabBuilder.classList.add("active-tab");
  }
}

function createBox(w, h, d, material) {
  const THREE = window.THREE;
  return new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
}

function updateHouseRangeLabels() {
  floorsValue.textContent = floorsInput.value;
  sectionsValue.textContent = sectionsInput.value;
  widthValue.textContent = widthInput.value;
  depthValue.textContent = depthInput.value;
  balconyRateValue.textContent = Number(balconyRate.value).toFixed(2);
  windowColsValue.textContent = windowCols.value;
}

function frameHouseCamera(width = 18) {
  camera.position.set(Math.max(18, width * 1.1), 12, Math.max(18, width * 1.1));
  controls.target.set(0, 5, 0);
}

// ================= MODULE DATA =================
function getModuleFormData() {
  return {
    name: moduleName.value.trim() || "Unnamed Module",
    type: moduleType.value,
    source_text: moduleDescription.value.trim(),
    params: {
      color: moduleColor.value,
      width: parseFloat(moduleWidth.value) || 1,
      height: parseFloat(moduleHeight.value) || 1.2,
      depth: parseFloat(moduleDepth.value) || 0.2,
      has_window: moduleHasWindow.checked,
      has_door: moduleHasDoor.checked,
      has_balcony: moduleHasBalcony.checked,
      is_roof_piece: moduleIsRoofPiece.checked
    }
  };
}

function applyModuleParams(data) {
  if (!data) return;

  if (data.name) moduleName.value = data.name;
  if (data.type) moduleType.value = data.type;

  const p = data.params || data;

  if (p.color) moduleColor.value = p.color;
  if (p.width != null) moduleWidth.value = p.width;
  if (p.height != null) moduleHeight.value = p.height;
  if (p.depth != null) moduleDepth.value = p.depth;

  moduleHasWindow.checked = Boolean(p.has_window);
  moduleHasDoor.checked = Boolean(p.has_door);
  moduleHasBalcony.checked = Boolean(p.has_balcony);
  moduleIsRoofPiece.checked = Boolean(p.is_roof_piece);

  const formData = getModuleFormData();
  moduleJson.textContent = JSON.stringify(formData, null, 2);
  setPreviewJson(formData);
}

function renderModulePreview(data) {
  clearSceneMeshes();

  const THREE = window.THREE;
  const group = new THREE.Group();
  const p = data.params || data;

  const bodyMaterial = new THREE.MeshStandardMaterial({ color: p.color || "#8f8f8f" });
  const darkMaterial = new THREE.MeshStandardMaterial({ color: 0x222222 });
  const glassMaterial = new THREE.MeshStandardMaterial({ color: 0x32435d });
  const balconyMaterial = new THREE.MeshStandardMaterial({ color: 0x666666 });

  const width = p.width || 1;
  const height = p.height || 1.2;
  const depth = p.depth || 0.2;

  const body = createBox(width, height, depth, bodyMaterial);
  body.position.y = height / 2;
  group.add(body);

  if (p.has_window) {
    const win = createBox(width * 0.5, height * 0.4, 0.03, glassMaterial);
    win.position.set(0, height * 0.55, depth / 2 + 0.02);
    group.add(win);
  }

  if (p.has_door) {
    const door = createBox(width * 0.35, height * 0.55, 0.03, darkMaterial);
    door.position.set(0, height * 0.28, depth / 2 + 0.02);
    group.add(door);
  }

  if (p.has_balcony) {
    const balcony = createBox(width * 0.7, 0.12, 0.35, balconyMaterial);
    balcony.position.set(0, height * 0.35, depth / 2 + 0.18);
    group.add(balcony);
  }

  if (p.is_roof_piece) {
    const roof = createBox(width + 0.1, 0.1, depth + 0.1, darkMaterial);
    roof.position.set(0, height + 0.05, 0);
    group.add(roof);
  }

  currentMesh = group;
  scene.add(group);

  camera.position.set(0, 1.5, 4);
  controls.target.set(0, 0.8, 0);
}

function renderSavedModules() {
  savedModules.innerHTML = "";

  const filterValue = libraryFilter.value || "all";
  const filtered = filterValue === "all"
    ? savedModulesData
    : savedModulesData.filter(m => m.type === filterValue);

  if (!filtered.length) {
    savedModules.innerHTML = `<div class="note">No saved modules yet</div>`;
    return;
  }

  filtered.forEach(mod => {
    const card = document.createElement("div");
    card.className = "module-card";
    card.innerHTML = `
      <div><strong>${mod.name}</strong></div>
      <div class="note">Type: ${mod.type}</div>
      <div class="module-actions">
        <button data-id="${mod.id}" class="small-btn use-module-btn">Edit</button>
        <button data-id="${mod.id}" class="small-btn use-in-builder-btn">Use in Builder</button>
        <button data-id="${mod.id}" class="small-btn delete-module-btn">Delete</button>
      </div>
    `;
    savedModules.appendChild(card);
  });

  document.querySelectorAll(".use-module-btn").forEach(btn => {
    btn.onclick = () => {
      const mod = savedModulesData.find(m => m.id === btn.dataset.id);
      if (!mod) return;
      applyModuleParams(mod);
      renderModulePreview(mod);
      setActiveTab("modules");
    };
  });

  document.querySelectorAll(".use-in-builder-btn").forEach(btn => {
    btn.onclick = () => {
      const mod = savedModulesData.find(m => m.id === btn.dataset.id);
      if (!mod) return;

      if (mod.type === "wall") wallModule.value = mod.id;
      if (mod.type === "window") windowModule.value = mod.id;
      if (mod.type === "door") doorModule.value = mod.id;
      if (mod.type === "roof") roofModule.value = mod.id;
      if (mod.type === "balcony") balconyModule.value = mod.id;

      const payload = getHouseFormData();
      houseJson.textContent = JSON.stringify(payload, null, 2);
      setPreviewJson(payload);
      setActiveTab("builder");
    };
  });

  document.querySelectorAll(".delete-module-btn").forEach(btn => {
    btn.onclick = async () => {
      const id = btn.dataset.id;

      try {
        await fetch(`${SERVER_URL}/api/modules/${id}`, { method: "DELETE" });
      } catch {}

      savedModulesData = savedModulesData.filter(m => m.id !== id);
      renderSavedModules();
      populateModuleSelectors();
    };
  });
}

async function loadSavedModules() {
  try {
    const r = await fetch(`${SERVER_URL}/api/modules`);
    const data = await r.json();
    savedModulesData = Array.isArray(data) ? data : [];
  } catch {
    savedModulesData = [];
  }

  renderSavedModules();
  populateModuleSelectors();
}

function populateModuleSelectors() {
  const byType = type => savedModulesData.filter(m => m.type === type);

  populateSelectWithModules(wallModule, byType("wall"));
  populateSelectWithModules(windowModule, byType("window"));
  populateSelectWithModules(doorModule, byType("door"));
  populateSelectWithModules(roofModule, byType("roof"));
  populateSelectWithModules(balconyModule, byType("balcony"));
}

function populateSelectWithModules(select, modules) {
  select.innerHTML = "";

  if (!modules.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No modules";
    select.appendChild(opt);
    return;
  }

  modules.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.name;
    select.appendChild(opt);
  });
}

// ================= HOUSE DATA =================
function getHouseFormData() {
  return {
    house: {
      floors: parseInt(floorsInput.value, 10),
      sections: parseInt(sectionsInput.value, 10),
      width: parseInt(widthInput.value, 10),
      depth: parseInt(depthInput.value, 10),
      roof_type: roofType.value,
      color: houseColor.value,
      has_balconies: hasBalconies.checked,
      balcony_rate: parseFloat(balconyRate.value),
      window_cols: parseInt(windowCols.value, 10)
    },
    modules: {
      wall: wallModule.value || null,
      window: windowModule.value || null,
      door: doorModule.value || null,
      roof: roofModule.value || null,
      balcony: balconyModule.value || null
    }
  };
}

function applyHouseParams(data) {
  const house = data.house || data;

  if (house.floors != null) floorsInput.value = house.floors;
  if (house.sections != null) sectionsInput.value = house.sections;
  if (house.width != null) widthInput.value = house.width;
  if (house.depth != null) depthInput.value = house.depth;
  if (house.roof_type) roofType.value = house.roof_type;
  if (house.color) houseColor.value = house.color;
  if (house.has_balconies != null) hasBalconies.checked = Boolean(house.has_balconies);
  if (house.balcony_rate != null) balconyRate.value = house.balcony_rate;
  if (house.window_cols != null) windowCols.value = house.window_cols;

  updateHouseRangeLabels();
  const formData = getHouseFormData();
  houseJson.textContent = JSON.stringify(formData, null, 2);
  setPreviewJson(formData);
}

function getModuleById(id) {
  return savedModulesData.find(m => m.id === id) || null;
}

// ================= HOUSE RENDER =================
function renderHousePreview(payload, offsetX = 0) {
  const THREE = window.THREE;
  const group = new THREE.Group();

  const house = { ...payload.house };

  house.floors = Math.max(3, Math.min(25, house.floors || 9));
  house.width = Math.max(8, Math.min(30, house.width || 16));
  house.depth = Math.max(1, Math.min(6, house.depth || 2));
  house.sections = Math.max(1, Math.min(house.sections || 3, Math.max(1, Math.floor(house.width / 4))));
  house.window_cols = Math.max(2, Math.min(house.window_cols || 8, house.width));
  house.balcony_rate = Math.max(0, Math.min(house.balcony_rate || 0.25, 0.35));
  house.roof_type = house.roof_type === "gable" ? "gable" : "flat";

  const floors = house.floors;
  const sections = house.sections;
  const width = house.width;
  const depth = house.depth;
  const floorHeight = 1.2;
  const buildingHeight = floors * floorHeight;

  const wallData = getModuleById(payload.modules.wall);
  const windowData = getModuleById(payload.modules.window);
  const doorData = getModuleById(payload.modules.door);
  const roofData = getModuleById(payload.modules.roof);
  const balconyData = getModuleById(payload.modules.balcony);

  const baseColor = new THREE.Color(house.color || "#8f8f8f");
  const panelColor = baseColor.clone().multiplyScalar(1.08);

  const bodyMaterial = new THREE.MeshStandardMaterial({ color: baseColor });
  const panelMaterial = new THREE.MeshStandardMaterial({ color: panelColor });
  const windowMaterial = new THREE.MeshStandardMaterial({ color: 0x32435d });
  const doorMaterial = new THREE.MeshStandardMaterial({ color: 0x222222 });
  const balconyMaterial = new THREE.MeshStandardMaterial({ color: 0x666666 });
  const roofMaterial = new THREE.MeshStandardMaterial({
    color: roofData?.params?.color || 0x444444
  });

  const body = createBox(width, buildingHeight, depth, bodyMaterial);
  body.position.set(offsetX, buildingHeight / 2, 0);
  group.add(body);

  // ===== Карта фасада =====
  const facade = [];
  for (let floor = 0; floor < floors; floor++) {
    facade[floor] = [];
    for (let col = 0; col < width; col++) {
      facade[floor][col] = {
        hasDoor: false,
        hasWindow: false,
        hasBalcony: false,
        sectionIndex: -1
      };
    }
  }

  // ===== Секции и двери =====
  const entranceCols = [];
  for (let s = 0; s < sections; s++) {
    const centerCol = Math.round(((s + 0.5) * width) / sections - 0.5);
    entranceCols.push(Math.max(0, Math.min(width - 1, centerCol)));
  }

  for (let i = 0; i < entranceCols.length; i++) {
    const col = entranceCols[i];
    facade[0][col].hasDoor = true;
    facade[0][col].sectionIndex = i;
  }

  // ===== Окна =====
  for (let floor = 0; floor < floors; floor++) {
    for (let col = 0; col < width; col++) {
      if (facade[floor][col].hasDoor) continue;

      const isEdge = col === 0 || col === width - 1;
      if (isEdge && Math.random() < 0.25) continue;

      facade[floor][col].hasWindow = true;
    }
  }

  // ===== Балконы только там, где уже есть окно =====
  if (house.has_balconies) {
    for (let floor = 1; floor < floors; floor++) {
      for (let col = 0; col < width; col++) {
        if (!facade[floor][col].hasWindow) continue;
        if (Math.random() < house.balcony_rate) {
          facade[floor][col].hasBalcony = true;
        }
      }
    }
  }

  // ===== Панели =====
  for (let floor = 0; floor < floors; floor++) {
    for (let col = 0; col < width; col++) {
      const panel = createBox(0.95, floorHeight * 0.95, 0.06, panelMaterial);
      panel.position.set(
        offsetX + col - width / 2 + 0.5,
        floor * floorHeight + floorHeight / 2,
        depth / 2 + 0.04
      );

      panel.userData.baseColor = panelMaterial.color.clone();
      panel.userData.label = `Floor ${floor + 1}`;
      interactiveObjects.push(panel);
      group.add(panel);
    }
  }

  // ===== Вертикальные швы секций =====
  for (let s = 1; s < sections; s++) {
    const seamX = offsetX - width / 2 + (s * width / sections);
    const seam = createBox(
      0.08,
      buildingHeight,
      0.08,
      new THREE.MeshStandardMaterial({ color: 0x7d7d7d })
    );
    seam.position.set(seamX, buildingHeight / 2, depth / 2 + 0.05);
    group.add(seam);
  }

  // ===== Подъездные зоны =====
  for (let i = 0; i < entranceCols.length; i++) {
    const col = entranceCols[i];
    const x = offsetX + col - width / 2 + 0.5;

    const entranceFrame = createBox(
      1.05,
      1.25,
      0.03,
      new THREE.MeshStandardMaterial({ color: 0x4c4c4c })
    );
    entranceFrame.position.set(x, 0.62, depth / 2 + 0.045);
    group.add(entranceFrame);
  }

  // ===== Окна и двери =====
  for (let floor = 0; floor < floors; floor++) {
    for (let col = 0; col < width; col++) {
      const cell = facade[floor][col];
      const x = offsetX + col - width / 2 + 0.5;
      const yCenter = floor * floorHeight + floorHeight / 2;

      if (cell.hasDoor) {
        const door = createBox(
          doorData?.params?.width || 0.55,
          doorData?.params?.height || 0.9,
          0.05,
          doorMaterial
        );
        door.position.set(x, 0.45, depth / 2 + 0.08);

        door.userData.baseColor = doorMaterial.color.clone();
        door.userData.label = `Entrance ${(cell.sectionIndex ?? 0) + 1}`;
        interactiveObjects.push(door);
        group.add(door);

        continue;
      }

      if (cell.hasWindow) {
        const win = createBox(
          windowData?.params?.width || 0.42,
          windowData?.params?.height || 0.42,
          0.05,
          windowMaterial
        );
        win.position.set(x, yCenter, depth / 2 + 0.08);
        group.add(win);
      }
    }
  }

  // ===== Балконы =====
  for (let floor = 1; floor < floors; floor++) {
    for (let col = 0; col < width; col++) {
      const cell = facade[floor][col];
      if (!cell.hasBalcony) continue;

      const x = offsetX + col - width / 2 + 0.5;
      const y = floor * floorHeight + 0.2;

      const balcony = createBox(
        balconyData?.params?.width || 0.72,
        0.18,
        balconyData?.params?.depth || 0.5,
        balconyMaterial
      );
      balcony.position.set(x, y, depth / 2 + 0.28);
      group.add(balcony);

      const railing = createBox(
        balconyData?.params?.width || 0.72,
        0.28,
        0.03,
        balconyMaterial
      );
      railing.position.set(x, y + 0.12, depth / 2 + 0.54);
      group.add(railing);
    }
  }

  // ===== Крыша =====
  if (house.roof_type === "gable") {
    const roofLeft = createBox(width + 0.2, 0.18, depth / 2 + 0.35, roofMaterial);
    roofLeft.rotation.x = Math.PI / 10;
    roofLeft.position.set(offsetX, buildingHeight + 0.28, -0.12);
    group.add(roofLeft);

    const roofRight = createBox(width + 0.2, 0.18, depth / 2 + 0.35, roofMaterial);
    roofRight.rotation.x = -Math.PI / 10;
    roofRight.position.set(offsetX, buildingHeight + 0.28, 0.12);
    group.add(roofRight);
  } else {
    const roof = createBox(
      width + 0.2,
      roofData?.params?.height || 0.22,
      depth + 0.2,
      roofMaterial
    );
    roof.position.set(offsetX, buildingHeight + 0.11, 0);
    group.add(roof);
  }

  return group;
}

function generateHousePreview() {
  clearSceneMeshes();

  const payload = getHouseFormData();
  currentMesh = renderHousePreview(payload, 0);
  scene.add(currentMesh);

  houseJson.textContent = JSON.stringify(payload, null, 2);
  setPreviewJson(payload);
  frameHouseCamera(payload.house.width);
}

async function animateHousePreview() {
  clearSceneMeshes();

  const payload = getHouseFormData();
  const house = payload.house;
  const THREE = window.THREE;

  const group = new THREE.Group();
  currentMesh = group;
  scene.add(group);

  const floors = Math.max(3, house.floors || 9);
  const width = Math.max(8, house.width || 16);
  const depth = Math.max(1, house.depth || 2);
  const floorHeight = 1.2;
  const buildingHeight = floors * floorHeight;

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  const body = createBox(
    width,
    buildingHeight,
    depth,
    new THREE.MeshStandardMaterial({ color: house.color || "#8f8f8f" })
  );
  body.position.set(0, buildingHeight / 2, 0);
  group.add(body);
  await sleep(180);

  for (let floor = 0; floor < floors; floor++) {
    const slab = createBox(
      width + 0.1,
      0.03,
      depth + 0.1,
      new THREE.MeshStandardMaterial({ color: 0x999999 })
    );
    slab.position.set(0, floor * floorHeight + floorHeight, 0);
    group.add(slab);
    await sleep(70);
  }

  clearMesh(group);
  currentMesh = renderHousePreview(payload, 0);
  scene.add(currentMesh);

  houseJson.textContent = JSON.stringify(payload, null, 2);
  setPreviewJson(payload);
  frameHouseCamera(payload.house.width);
}

function compareHousePreview() {
  if (!currentMesh) return;

  if (compareMesh) clearMesh(compareMesh);

  const payload = getHouseFormData();
  payload.house.floors = Math.min(25, payload.house.floors + 2);
  payload.house.color = "#707b86";

  compareMesh = renderHousePreview(payload, payload.house.width + 10);
  scene.add(compareMesh);

  camera.position.set(30, 16, 28);
  controls.target.set(payload.house.width / 2 + 4, 5, 0);
}

// ================= ZIP HELPERS =================
async function loadZipJson(url, expectedFile) {
  if (url.startsWith("/")) url = SERVER_URL + url;

  const res = await fetch(url);
  const arrayBuf = await res.arrayBuffer();
  latestExportZipBlob = new Blob([arrayBuf], { type: "application/zip" });

  const zip = await JSZip.loadAsync(arrayBuf);
  const file = zip.file(expectedFile);

  if (!file) {
    throw new Error(`No ${expectedFile} in zip`);
  }

  const text = await file.async("text");
  return JSON.parse(text);
}

// ================= EVENTS =================
tabModules.onclick = () => setActiveTab("modules");
tabLibrary.onclick = () => setActiveTab("library");
tabBuilder.onclick = () => setActiveTab("builder");

libraryFilter.addEventListener("change", renderSavedModules);

analyzeModuleBtn.onclick = async () => {
  const text = moduleDescription.value.trim();
  if (!text) {
    alert("Введите описание модуля");
    return;
  }

  analyzeModuleBtn.disabled = true;
  analyzeModuleBtn.textContent = "Analyzing...";

  try {
    const r = await fetch(`${SERVER_URL}/api/analyze-module-text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await r.json();
    if (!data.zip_url) throw new Error("No zip returned");

    const parsed = await loadZipJson(data.zip_url, "module_params.json");
    applyModuleParams(parsed);
    renderModulePreview(getModuleFormData());
  } catch (err) {
    alert("Analyze failed: " + err.message);
  }

  analyzeModuleBtn.disabled = false;
  analyzeModuleBtn.textContent = "Analyze";
};

generateModuleBtn.onclick = () => {
  const data = getModuleFormData();
  moduleJson.textContent = JSON.stringify(data, null, 2);
  setPreviewJson(data);
  renderModulePreview(data);
};

saveModuleBtn.onclick = async () => {
  const data = getModuleFormData();

  try {
    const r = await fetch(`${SERVER_URL}/api/save-module`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    const result = await r.json().catch(() => ({}));
    const id = result.module_id || `module_${Date.now()}`;
    savedModulesData.push({ ...data, id });
  } catch {
    savedModulesData.push({ ...data, id: `module_${Date.now()}` });
  }

  renderSavedModules();
  populateModuleSelectors();
  setActiveTab("library");
};

analyzeHouseBtn.onclick = async () => {
  const text = houseDescription.value.trim();
  if (!text) {
    alert("Введите описание дома");
    return;
  }

  analyzeHouseBtn.disabled = true;
  analyzeHouseBtn.textContent = "Analyzing...";

  try {
    const r = await fetch(`${SERVER_URL}/api/analyze-building-text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await r.json();
    if (!data.zip_url) throw new Error("No zip returned");

    const parsed = await loadZipJson(data.zip_url, "building_params.json");
    applyHouseParams(parsed);
    generateHousePreview();
  } catch (err) {
    alert("Analyze failed: " + err.message);
  }

  analyzeHouseBtn.disabled = false;
  analyzeHouseBtn.textContent = "Analyze";
};

generateHouseBtn.onclick = generateHousePreview;
animateHouseBtn.onclick = animateHousePreview;
compareHouseBtn.onclick = compareHousePreview;

exportZipBtn.onclick = async () => {
  const payload = getHouseFormData();

  try {
    const r = await fetch(`${SERVER_URL}/api/export-house-zip`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await r.json();
    if (!data.zip_url) throw new Error("No zip returned");

    const res = await fetch(`${SERVER_URL}${data.zip_url}`);
    latestExportZipBlob = await res.blob();

    const a = document.createElement("a");
    a.href = URL.createObjectURL(latestExportZipBlob);
    a.download = "house_export.zip";
    a.click();
  } catch (err) {
    alert("Export failed: " + err.message);
  }
};

resetBtn.onclick = () => {
  moduleDescription.value = "";
  moduleName.value = "";
  moduleType.value = "wall";
  moduleColor.value = "#8f8f8f";
  moduleWidth.value = 1;
  moduleHeight.value = 1.2;
  moduleDepth.value = 0.2;
  moduleHasWindow.checked = false;
  moduleHasDoor.checked = false;
  moduleHasBalcony.checked = false;
  moduleIsRoofPiece.checked = false;
  moduleJson.textContent = "{}";

  houseDescription.value = "";
  floorsInput.value = 8;
  sectionsInput.value = 3;
  widthInput.value = 18;
  depthInput.value = 2;
  houseColor.value = "#8f8f8f";
  roofType.value = "flat";
  hasBalconies.checked = true;
  balconyRate.value = 0.25;
  windowCols.value = 8;
  updateHouseRangeLabels();

  houseJson.textContent = "{}";
  previewJson.textContent = "{}";

  clearSceneMeshes();
  camera.position.set(18, 12, 18);
  controls.target.set(0, 4, 0);
};

[
  floorsInput,
  sectionsInput,
  widthInput,
  depthInput,
  balconyRate,
  windowCols
].forEach(input => {
  input.addEventListener("input", () => {
    updateHouseRangeLabels();
    const payload = getHouseFormData();
    houseJson.textContent = JSON.stringify(payload, null, 2);
    setPreviewJson(payload);
  });
});

[
  houseColor,
  roofType,
  hasBalconies,
  wallModule,
  windowModule,
  doorModule,
  roofModule,
  balconyModule
].forEach(input => {
  input.addEventListener("input", () => {
    const payload = getHouseFormData();
    houseJson.textContent = JSON.stringify(payload, null, 2);
    setPreviewJson(payload);
  });

  input.addEventListener("change", () => {
    const payload = getHouseFormData();
    houseJson.textContent = JSON.stringify(payload, null, 2);
    setPreviewJson(payload);
  });
});

function onPointerMove(event) {
  if (!interactiveObjects.length) return;

  const rect = preview.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(interactiveObjects, false);

  interactiveObjects.forEach(obj => {
    if (obj.userData.baseColor) {
      obj.material.color.copy(obj.userData.baseColor);
    }
  });

  if (intersects.length > 0) {
    const hit = intersects[0].object;
    hit.material.color.set(0xb87cff);
    hoverInfo.classList.remove("hidden");
    hoverInfo.textContent = hit.userData.label || "Element";
  } else {
    hoverInfo.classList.add("hidden");
  }
}

// ================= INIT =================
(async () => {
  await initScene();
  updateHouseRangeLabels();
  await loadSavedModules();
  setActiveTab("modules");

  floorsInput.value = 8;
  sectionsInput.value = 3;
  widthInput.value = 18;
  depthInput.value = 2;
  houseColor.value = "#8f8f8f";
  roofType.value = "flat";
  hasBalconies.checked = true;
  balconyRate.value = 0.25;
  windowCols.value = 8;
  updateHouseRangeLabels();

  const initialModule = getModuleFormData();
  moduleJson.textContent = JSON.stringify(initialModule, null, 2);
  setPreviewJson(initialModule);
  renderModulePreview(initialModule);
})
();