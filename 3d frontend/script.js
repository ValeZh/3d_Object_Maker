const SERVER_URL = location.origin;

// ================= CONSTANTS =================
const DEFAULTS = {
  module: {
    wall: { width: 1, height: 1.2, depth: 0.2 },
    window: { width: 0.45, height: 0.45, depth: 0.05 },
    door: { width: 0.55, height: 0.9, depth: 0.05 },
    roof: { width: 1, height: 0.28, depth: 0.28 },
    balcony: { width: 0.72, height: 0.18, depth: 0.5 }
  },
  normalize: {
    wall: { widthMin: 0.7, widthMax: 0.98, heightMin: 0.85, heightMax: 1.18, depthMin: 0.04, depthMax: 0.18 },
    window: { widthMin: 0.25, widthMax: 0.55, heightMin: 0.25, heightMax: 0.55, depth: 0.05 },
    door: { widthMin: 0.45, widthMax: 0.8, heightMin: 0.8, heightMax: 1.2, depth: 0.05 },
    balcony: { widthMin: 0.5, widthMax: 1.0, depthMin: 0.25, depthMax: 0.8 }
  },
  house: {
    minFloors: 3,
    maxFloors: 25,
    minWidth: 8,
    maxWidth: 30,
    minDepth: 1,
    maxDepth: 6,
    minSections: 1,
    maxSections: 10,
    floorHeight: 1.2,
    minWindowCols: 2,
    roofType: "flat"
  },
  colors: {
    wall: "#8f8f8f",
    window: "#32435d",
    door: "#2a2a2a",
    balcony: "#666666",
    roof: "#444444",
    seam: 0x7d7d7d,
    entranceFrame: 0x4c4c4c
  },
  roof: {
    flatOverhang: 0.3,
    minThickness: 0.12
  },
  facade: {
    textureScale: 3,
    balconyMaxRate: 0.6,
    blindPanelChance: 0.12,
    topFloorBlindBoost: 0.08,
    edgeReductionChance: 0.15,
    sectionShiftMax: 2
  },
  camera: {
    padding: 1.45,
    minDistance: 6
  },
  randomize: {
    floors: [5, 18],
    sections: [1, 6],
    width: [10, 26],
    depth: [1, 5],
    balconyRate: [0.1, 0.45],
    windowCols: [3, 14],
    textureScale: [1, 8]
  },
  textParsing: {
    colors: {
      gray: "#8f8f8f",
      grey: "#8f8f8f",
      white: "#d9d9d9",
      black: "#2a2a2a",
      red: "#c74a4a",
      blue: "#4a6fc7",
      green: "#4aa36c",
      beige: "#c9b28f",
      brown: "#8b6a4e"
    }
  }
};

// ================= DOM HELPERS =================
const $ = id => document.getElementById(id);

// ================= UI TABS =================
const tabModules = $("tabModules");
const tabLibrary = $("tabLibrary");
const tabBuilder = $("tabBuilder");

const modulePage = $("modulePage");
const libraryPage = $("libraryPage");
const builderPage = $("builderPage");

const resetBtn = $("reset");

// ================= ADVANCED TOGGLES =================
const moduleAdvancedToggle = $("moduleAdvancedToggle");
const moduleAdvancedContent = $("moduleAdvancedContent");
const houseAdvancedToggle = $("houseAdvancedToggle");
const houseAdvancedContent = $("houseAdvancedContent");

// ================= MODULE GENERATOR UI =================
const moduleDescription = $("moduleDescription");
const analyzeModuleBtn = $("analyzeModule");
const moduleName = $("moduleName");
const moduleType = $("moduleType");
const moduleColor = $("moduleColor");
const moduleWidth = $("moduleWidth");
const moduleHeight = $("moduleHeight");
const moduleDepth = $("moduleDepth");
const generateModuleBtn = $("generateModule");
const saveModuleBtn = $("saveModule");

// ================= MODULE LIBRARY UI =================
const libraryFilter = $("libraryFilter");
const librarySort = $("librarySort");
const savedModules = $("savedModules");
const exportLibraryZipBtn = $("exportLibraryZipBtn");
const importLibraryZipBtn = $("importLibraryZipBtn");
const importLibraryZipInput = $("importLibraryZipInput");

// ================= HOUSE BUILDER UI =================
const houseDescription = $("houseDescription");
const analyzeHouseBtn = $("analyzeHouse");

const floorsInput = $("floors");
const sectionsInput = $("sections");
const widthInput = $("width");
const depthInput = $("depth");
const facadeTextureSelect = $("facadeTextureSelect");
const facadeTextureRepeat = $("facadeTextureRepeat");
const hasBalconies = $("hasBalconies");
const balconyRate = $("balconyRate");
const windowCols = $("windowCols");

const floorsValue = $("floorsValue");
const sectionsValue = $("sectionsValue");
const widthValue = $("widthValue");
const depthValue = $("depthValue");
const balconyRateValue = $("balconyRateValue");
const windowColsValue = $("windowColsValue");
const facadeTextureRepeatValue = $("facadeTextureRepeatValue");

const wallModule = $("wallModule");
const windowModule = $("windowModule");
const doorModule = $("doorModule");
const roofModule = $("roofModule");
const balconyModule = $("balconyModule");

const generateHouseBtn = $("generateHouse");
const animateHouseBtn = $("animateHouse");
const randomizeHouseBtn = $("randomizeHouseBtn");
const randomizeFacadeBtn = $("randomizeFacadeBtn");

const houseName = $("houseName");
const saveHouseBtn = $("saveHouseBtn");
const exportProjectZipBtn = $("exportProjectZipBtn");
const importProjectZipBtn = $("importProjectZipBtn");
const importProjectZipInput = $("importProjectZipInput");
const houseSort = $("houseSort");
const savedHouses = $("savedHouses");

// ================= ERRORS =================
const moduleFormError = $("moduleFormError");
const houseFormError = $("houseFormError");

// ================= PREVIEW =================
const preview = $("objectPreview");
const previewJson = $("previewJson");
const hoverInfo = $("hoverInfo");
const toastContainer = $("toastContainer");

// ================= STATE =================
let scene, camera, renderer, controls;
let currentMesh = null;
let groundPlane = null;
let savedModulesData = [];
let savedHousesData = [];
let interactiveObjects = [];
let raycaster, mouse;
let lastHovered = null;

const textureCache = new Map();
const geometryCache = new Map();
const sharedMaterialCache = new Map();

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
    2000
  );
  camera.position.set(18, 12, 18);

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    preserveDrawingBuffer: true
  });
  renderer.setSize(preview.clientWidth, preview.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  preview.appendChild(renderer.domElement);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x333333, 1.15);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 1.15);
  dir.position.set(18, 28, 14);
  dir.castShadow = true;
  dir.shadow.mapSize.width = 2048;
  dir.shadow.mapSize.height = 2048;
  dir.shadow.camera.near = 0.5;
  dir.shadow.camera.far = 120;
  dir.shadow.camera.left = -50;
  dir.shadow.camera.right = 50;
  dir.shadow.camera.top = 50;
  dir.shadow.camera.bottom = -50;
  scene.add(dir);

  const grid = new THREE.GridHelper(120, 120, 0x333333, 0x222222);
  scene.add(grid);

  const groundMat = new THREE.ShadowMaterial({ opacity: 0.18 });
  groundMat.userData.shared = true;
  groundPlane = new THREE.Mesh(new THREE.PlaneGeometry(180, 180), groundMat);
  groundPlane.rotation.x = -Math.PI / 2;
  groundPlane.position.y = 0.001;
  groundPlane.receiveShadow = true;
  scene.add(groundPlane);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 4, 0);

  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  preview.addEventListener("mousemove", onPointerMove);
  preview.addEventListener("mouseleave", handlePointerLeave);
  window.addEventListener("resize", handleResize);

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls?.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
}

function handleResize() {
  if (!camera || !renderer) return;
  camera.aspect = preview.clientWidth / preview.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(preview.clientWidth, preview.clientHeight);
}

// ================= GENERIC HELPERS =================
function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function randInt(min, max) {
  return Math.floor(min + Math.random() * (max - min + 1));
}

function randFloat(min, max, step = null) {
  const raw = min + Math.random() * (max - min);
  if (!step) return raw;
  return Math.round(raw / step) * step;
}

function sanitizeFilename(name, fallback = "file") {
  return String(name || fallback)
    .trim()
    .replace(/[<>:"/\\|?*]+/g, "_")
    .replace(/\s+/g, "_")
    .slice(0, 80) || fallback;
}

async function nextFrame() {
  return new Promise(resolve => requestAnimationFrame(() => resolve()));
}

function getRendererPreviewBase64() {
  if (!renderer?.domElement) throw new Error("Renderer is not ready");
  const dataUrl = renderer.domElement.toDataURL("image/png");
  return dataUrl.split(",")[1];
}

function sortItems(items, sortBy) {
  const result = [...items];

  const compareDate = (a, b, dir = "desc") => {
    const av = new Date(a.created_at || 0).getTime();
    const bv = new Date(b.created_at || 0).getTime();
    return dir === "desc" ? bv - av : av - bv;
  };

  const compareName = (a, b, dir = "asc") => {
    const av = String(a.name || "").localeCompare(String(b.name || ""), undefined, { sensitivity: "base" });
    return dir === "asc" ? av : -av;
  };

  const compareType = (a, b, dir = "asc") => {
    const av = String(a.type || "").localeCompare(String(b.type || ""), undefined, { sensitivity: "base" });
    if (av !== 0) return dir === "asc" ? av : -av;
    return compareName(a, b, "asc");
  };

  if (sortBy === "date_desc") result.sort((a, b) => compareDate(a, b, "desc"));
  if (sortBy === "date_asc") result.sort((a, b) => compareDate(a, b, "asc"));
  if (sortBy === "name_asc") result.sort((a, b) => compareName(a, b, "asc"));
  if (sortBy === "name_desc") result.sort((a, b) => compareName(a, b, "desc"));
  if (sortBy === "type_asc") result.sort((a, b) => compareType(a, b, "asc"));
  if (sortBy === "type_desc") result.sort((a, b) => compareType(a, b, "desc"));

  return result;
}

function showToast(message, type = "info", title = "") {
  if (!toastContainer) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <div class="toast-title">${title || (type === "success" ? "Success" : type === "error" ? "Error" : "Info")}</div>
    <div class="toast-text">${message}</div>
  `;
  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(-8px)";
    setTimeout(() => toast.remove(), 220);
  }, 3200);
}

async function withLoading(button, loadingText, action) {
  if (!button) return action();

  const originalText = button.textContent;
  button.disabled = true;
  button.classList.add("loading");
  button.textContent = loadingText;

  try {
    return await action();
  } finally {
    button.disabled = false;
    button.classList.remove("loading");
    button.textContent = originalText;
  }
}

function setFieldError(input, message = "", errorId = null) {
  if (input) {
    input.classList.toggle("invalid-field", Boolean(message));
  }
  if (errorId) {
    const el = $(errorId);
    if (el) {
      el.textContent = message || "";
      el.classList.toggle("hidden", !message);
    }
  }
}

// ================= LOW LEVEL 3D HELPERS =================
function getBoxGeometry(w, h, d) {
  const key = `${w}|${h}|${d}`;
  if (!geometryCache.has(key)) {
    const g = new window.THREE.BoxGeometry(w, h, d);
    g.userData.cached = true;
    geometryCache.set(key, g);
  }
  return geometryCache.get(key);
}

function createBox(w, h, d, material) {
  const mesh = new window.THREE.Mesh(getBoxGeometry(w, h, d), material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function getSharedMaterial(key, factory) {
  if (!sharedMaterialCache.has(key)) {
    const material = factory();
    material.userData.shared = true;
    material.userData.disposed = false;
    sharedMaterialCache.set(key, material);
  }
  return sharedMaterialCache.get(key);
}

function createMaterialVariant(baseMaterial, colorOverride = null) {
  const material = baseMaterial.clone();
  material.userData.shared = false;
  material.userData.disposed = false;
  if (colorOverride != null) {
    material.color = new window.THREE.Color(colorOverride);
  }
  return material;
}

function isDisposableGeometry(geometry) {
  return geometry && !geometry.userData?.cached && !geometry.userData?.disposed;
}

function isDisposableMaterial(material) {
  return material && !material.userData?.shared && !material.userData?.disposed;
}

function disposeGeometrySafe(geometry) {
  if (!isDisposableGeometry(geometry)) return;
  geometry.dispose?.();
  geometry.userData.disposed = true;
}

function disposeMaterialSafe(material) {
  if (!isDisposableMaterial(material)) return;
  material.dispose?.();
  material.userData.disposed = true;
}

function clearHoverState() {
  if (lastHovered?.object && lastHovered.baseColor) {
    lastHovered.object.material.color.copy(lastHovered.baseColor);
  }
  lastHovered = null;
}

function clearMesh(mesh) {
  if (!mesh) return;

  scene.remove(mesh);

  mesh.traverse(obj => {
    if (obj.geometry) disposeGeometrySafe(obj.geometry);

    if (obj.material) {
      if (Array.isArray(obj.material)) obj.material.forEach(disposeMaterialSafe);
      else disposeMaterialSafe(obj.material);
    }
  });
}

function clearSceneMeshes() {
  clearHoverState();

  if (currentMesh) clearMesh(currentMesh);

  currentMesh = null;
  interactiveObjects = [];

  const objectsToRemove = [];
  scene.children.forEach((child) => {
    if (child !== groundPlane && !(child instanceof THREE.GridHelper)) {
      objectsToRemove.push(child);
    }
  });
  objectsToRemove.forEach(obj => scene.remove(obj));
}

function frameObject(mesh, padding = DEFAULTS.camera.padding) {
  if (!mesh || !camera || !controls) return;

  const THREE = window.THREE;
  const box = new THREE.Box3().setFromObject(mesh);
  if (box.isEmpty()) return;

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  const maxSize = Math.max(size.x, size.y, size.z, 1);
  const fov = camera.fov * (Math.PI / 180);
  let distance = (maxSize * padding) / (2 * Math.tan(fov / 2));
  distance = Math.max(distance, DEFAULTS.camera.minDistance);

  camera.position.set(
    center.x + distance * 0.95,
    center.y + distance * 0.58,
    center.z + distance * 0.95
  );
  controls.target.copy(center);
  controls.update();
}

// ================= PREVIEW / UI HELPERS =================
function setPreviewJson(data) {
  if (previewJson) previewJson.textContent = JSON.stringify(data, null, 2);
}

function updateModuleJsonPreview() {
  setPreviewJson(getModuleFormData());
}

function updateHouseJsonPreview() {
  setPreviewJson(getHouseFormData());
}

function setActiveTab(tab) {
  modulePage?.classList.add("hidden");
  libraryPage?.classList.add("hidden");
  builderPage?.classList.add("hidden");

  tabModules?.classList.remove("active-tab");
  tabLibrary?.classList.remove("active-tab");
  tabBuilder?.classList.remove("active-tab");

  if (tab === "modules") {
    modulePage?.classList.remove("hidden");
    tabModules?.classList.add("active-tab");
    liveUpdateModulePreview();
    return;
  }

  if (tab === "library") {
    libraryPage?.classList.remove("hidden");
    tabLibrary?.classList.add("active-tab");
    return;
  }

  builderPage?.classList.remove("hidden");
  tabBuilder?.classList.add("active-tab");
  updateHouseJsonPreview();
}

function paintRange(input) {
  if (!input) return;
  const min = Number(input.min || 0);
  const max = Number(input.max || 100);
  const value = Number(input.value || 0);
  const percent = ((value - min) / (max - min)) * 100;
  input.style.background =
    `linear-gradient(90deg, #2d8cff 0%, #2d8cff ${percent}%, #d9d9d9 ${percent}%, #d9d9d9 100%)`;
}

function updateHouseRangeLabels() {
  if (floorsValue) floorsValue.textContent = floorsInput.value;
  if (sectionsValue) sectionsValue.textContent = sectionsInput.value;
  if (widthValue) widthValue.textContent = widthInput.value;
  if (depthValue) depthValue.textContent = depthInput.value;
  if (balconyRateValue) balconyRateValue.textContent = Number(balconyRate.value).toFixed(2);
  if (windowColsValue) windowColsValue.textContent = windowCols.value;
  if (facadeTextureRepeatValue) facadeTextureRepeatValue.textContent = facadeTextureRepeat.value;

  [
    floorsInput,
    sectionsInput,
    widthInput,
    depthInput,
    balconyRate,
    windowCols,
    facadeTextureRepeat
  ].forEach(paintRange);
}

function toggleAdvanced(toggleEl, contentEl) {
  if (!toggleEl || !contentEl) return;
  contentEl.classList.toggle("open");
  toggleEl.classList.toggle("open");
}

// ================= TEXT PARSING HELPERS =================
function parseColorFromText(text) {
  const t = text.toLowerCase();
  for (const [word, hex] of Object.entries(DEFAULTS.textParsing.colors)) {
    if (new RegExp(`\\b${word}\\b`).test(t)) return hex;
  }

  const hexMatch = t.match(/#([0-9a-f]{6})\b/i);
  if (hexMatch) return `#${hexMatch[1]}`;
  return null;
}

function parseModuleTextLocally(text) {
  const t = text.toLowerCase();
  const result = {};

  if (/\bwindow\b/.test(t)) result.type = "window";
  else if (/\bdoor\b/.test(t)) result.type = "door";
  else if (/\broof\b/.test(t)) result.type = "roof";
  else if (/\bbalcony\b/.test(t)) result.type = "balcony";
  else if (/\bwall\b|\bpanel\b/.test(t)) result.type = "wall";

  const color = parseColorFromText(t);
  if (color) result.color = color;

  const widthMatch = t.match(/\bwidth\s*(\d+(\.\d+)?)\b/);
  const heightMatch = t.match(/\bheight\s*(\d+(\.\d+)?)\b/);
  const depthMatch = t.match(/\bdepth\s*(\d+(\.\d+)?)\b/);

  if (widthMatch) result.width = parseFloat(widthMatch[1]);
  if (heightMatch) result.height = parseFloat(heightMatch[1]);
  if (depthMatch) result.depth = parseFloat(depthMatch[1]);

  if (!result.width || !result.height || !result.depth) {
    if (/\bsmall\b/.test(t)) {
      result.width ??= 0.6;
      result.height ??= 0.6;
      result.depth ??= 0.08;
    }
    if (/\blarge\b|\bbig\b/.test(t)) {
      result.width ??= 1.4;
      result.height ??= 1.5;
      result.depth ??= 0.25;
    }
    if (/\bthin\b/.test(t)) {
      result.depth ??= 0.05;
    }
    if (/\bwide\b/.test(t)) {
      result.width ??= 1.4;
    }
    if (/\btall\b/.test(t)) {
      result.height ??= 1.6;
    }
  }

  if (!result.type && /\bentrance\b/.test(t)) result.type = "door";

  return result;
}

function parseHouseTextLocally(text) {
  const t = text.toLowerCase();
  const result = {
    roof_type: "flat"
  };

  const floorsMatch = t.match(/(\d+)\s*(floors|floor|storeys|stories)/);
  const sectionsMatch = t.match(/(\d+)\s*(sections|entrances|entries)/);
  const widthMatch = t.match(/\bwidth\s*(\d+)\b/);
  const depthMatch = t.match(/\bdepth\s*(\d+)\b/);
  const windowsMatch = t.match(/(\d+)\s*(window columns|window column|windows)/);

  if (floorsMatch) result.floors = parseInt(floorsMatch[1], 10);
  if (sectionsMatch) result.sections = parseInt(sectionsMatch[1], 10);
  if (widthMatch) result.width = parseInt(widthMatch[1], 10);
  if (depthMatch) result.depth = parseInt(depthMatch[1], 10);
  if (windowsMatch) result.window_cols = parseInt(windowsMatch[1], 10);

  if (/\bbalcony\b|\bbalconies\b/.test(t)) result.has_balconies = true;
  if (/\bno balconies\b|\bwithout balconies\b/.test(t)) result.has_balconies = false;

  if (/\blong\b/.test(t)) result.width ??= 24;
  if (/\bnarrow\b/.test(t)) result.depth ??= 1;
  if (/\bdeep\b/.test(t)) result.depth ??= 4;
  if (/\bcompact\b/.test(t)) {
    result.width ??= 12;
    result.floors ??= 6;
  }
  if (/\btall\b/.test(t)) result.floors ??= 14;
  if (/\blow-rise\b/.test(t)) result.floors ??= 5;

  const color = parseColorFromText(t);
  if (color) result.wall_color = color;

  const balconyRateMatch = t.match(/balcony\s*(density|rate)?\s*(\d+(\.\d+)?)/);
  if (balconyRateMatch) result.balcony_rate = clamp(parseFloat(balconyRateMatch[2]), 0, 1);

  return result;
}

function applyLocalModuleParse(parsed) {
  if (!parsed || typeof parsed !== "object") return;

  if (parsed.type && moduleType.value !== parsed.type) {
    moduleType.value = parsed.type;
    applyModuleTypeDefaults(parsed.type);
  }

  if (parsed.color) moduleColor.value = parsed.color;
  if (parsed.width != null) moduleWidth.value = parsed.width;
  if (parsed.height != null) moduleHeight.value = parsed.height;
  if (parsed.depth != null) moduleDepth.value = parsed.depth;

  updateModuleJsonPreview();
  renderModulePreview(getModuleFormData());
}

function applyLocalHouseParse(parsed) {
  if (!parsed || typeof parsed !== "object") return;

  if (parsed.floors != null) floorsInput.value = clamp(parsed.floors, 1, DEFAULTS.house.maxFloors);
  if (parsed.sections != null) sectionsInput.value = clamp(parsed.sections, 1, DEFAULTS.house.maxSections);
  if (parsed.width != null) widthInput.value = clamp(parsed.width, 6, DEFAULTS.house.maxWidth);
  if (parsed.depth != null) depthInput.value = clamp(parsed.depth, 1, DEFAULTS.house.maxDepth);
  if (parsed.window_cols != null) windowCols.value = clamp(parsed.window_cols, 2, parseInt(widthInput.value, 10));
  if (parsed.has_balconies != null) hasBalconies.checked = Boolean(parsed.has_balconies);
  if (parsed.balcony_rate != null) balconyRate.value = clamp(parsed.balcony_rate, 0, 1);

  updateHouseRangeLabels();
  updateHouseJsonPreview();
}

// ================= MODULE FORM HELPERS =================
function getModuleFlagsByType(type) {
  return {
    has_window: type === "window",
    has_door: type === "door",
    has_balcony: type === "balcony",
    is_roof_piece: type === "roof"
  };
}

function applyModuleTypeDefaults(type) {
  const defaults = DEFAULTS.module[type] || DEFAULTS.module.wall;
  moduleWidth.value = defaults.width;
  moduleHeight.value = defaults.height;
  moduleDepth.value = defaults.depth;
}

function getModuleFormData() {
  const flags = getModuleFlagsByType(moduleType.value);

  return {
    name: moduleName.value.trim() || "Unnamed Module",
    type: moduleType.value,
    source_text: moduleDescription.value.trim(),
    created_at: new Date().toISOString(),
    params: {
      color: moduleColor.value,
      width: parseFloat(moduleWidth.value) || DEFAULTS.module.wall.width,
      height: parseFloat(moduleHeight.value) || DEFAULTS.module.wall.height,
      depth: parseFloat(moduleDepth.value) || DEFAULTS.module.wall.depth,
      ...flags
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

  updateModuleJsonPreview();
}

function clearModuleValidation() {
  [
    [moduleName, "moduleNameError"],
    [moduleWidth, "moduleWidthError"],
    [moduleHeight, "moduleHeightError"],
    [moduleDepth, "moduleDepthError"]
  ].forEach(([input, id]) => setFieldError(input, "", id));

  moduleFormError?.classList.add("hidden");
  if (moduleFormError) moduleFormError.textContent = "";
}

function validateModuleForm(showErrors = true) {
  clearModuleValidation();

  const errors = [];
  const name = moduleName.value.trim();
  const type = moduleType.value;
  const width = parseFloat(moduleWidth.value);
  const height = parseFloat(moduleHeight.value);
  const depth = parseFloat(moduleDepth.value);

  if (!name) {
    errors.push("Module name is required.");
    if (showErrors) setFieldError(moduleName, "Enter a module name.", "moduleNameError");
  }

  if (!(width > 0)) {
    errors.push("Width must be greater than 0.");
    if (showErrors) setFieldError(moduleWidth, "Width must be greater than 0.", "moduleWidthError");
  }

  if (!(height > 0)) {
    errors.push("Height must be greater than 0.");
    if (showErrors) setFieldError(moduleHeight, "Height must be greater than 0.", "moduleHeightError");
  }

  if (!(depth > 0)) {
    errors.push("Depth must be greater than 0.");
    if (showErrors) setFieldError(moduleDepth, "Depth must be greater than 0.", "moduleDepthError");
  }

  if (type === "window" && (width > 2 || height > 2)) {
    errors.push("Window module is unusually large.");
    if (showErrors) {
      setFieldError(moduleWidth, "Window is too wide for a typical module.", "moduleWidthError");
      setFieldError(moduleHeight, "Window is too tall for a typical module.", "moduleHeightError");
    }
  }

  if (type === "door" && height < 0.7) {
    errors.push("Door height is too small.");
    if (showErrors) setFieldError(moduleHeight, "Door height should be at least 0.7.", "moduleHeightError");
  }

  if (type === "roof" && depth < 0.08) {
    errors.push("Roof depth is too small.");
    if (showErrors) setFieldError(moduleDepth, "Roof depth should be at least 0.08.", "moduleDepthError");
  }

  if (showErrors && errors.length) {
    moduleFormError.textContent = "Please fix the highlighted module fields.";
    moduleFormError.classList.remove("hidden");
  }

  return { valid: errors.length === 0, errors };
}

// ================= HOUSE FORM HELPERS =================
function getHouseFormData() {
  return {
    house: {
      floors: parseInt(floorsInput.value, 10),
      sections: parseInt(sectionsInput.value, 10),
      width: parseInt(widthInput.value, 10),
      depth: parseInt(depthInput.value, 10),
      roof_type: DEFAULTS.house.roofType,
      facade: {
        texture_url: facadeTextureSelect?.value || "",
        texture_scale: parseFloat(facadeTextureRepeat?.value || DEFAULTS.facade.textureScale)
      },
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
  const facade = house.facade || {};

  if (house.floors != null) floorsInput.value = house.floors;
  if (house.sections != null) sectionsInput.value = house.sections;
  if (house.width != null) widthInput.value = house.width;
  if (house.depth != null) depthInput.value = house.depth;

  const incomingTextureUrl = facade.texture_url || house.facade_texture_url || "";
  if (facadeTextureSelect) {
    if ([...facadeTextureSelect.options].some(opt => opt.value === incomingTextureUrl)) {
      facadeTextureSelect.value = incomingTextureUrl;
    } else {
      facadeTextureSelect.value = "";
    }
  }

  const incomingTextureScale = facade.texture_scale ?? house.texture_scale;
  if (incomingTextureScale != null && facadeTextureRepeat) {
    facadeTextureRepeat.value = incomingTextureScale;
  }

  if (house.has_balconies != null) hasBalconies.checked = Boolean(house.has_balconies);
  if (house.balcony_rate != null) balconyRate.value = house.balcony_rate;
  if (house.window_cols != null) windowCols.value = house.window_cols;

  if (data.modules) {
    if (data.modules.wall != null) wallModule.value = data.modules.wall || "";
    if (data.modules.window != null) windowModule.value = data.modules.window || "";
    if (data.modules.door != null) doorModule.value = data.modules.door || "";
    if (data.modules.roof != null) roofModule.value = data.modules.roof || "";
    if (data.modules.balcony != null) balconyModule.value = data.modules.balcony || "";
  }

  updateHouseRangeLabels();
  updateHouseJsonPreview();
}

function clearHouseValidation() {
  [
    [floorsInput, "floorsError"],
    [sectionsInput, "sectionsError"],
    [widthInput, "widthError"],
    [depthInput, "depthError"],
    [balconyRate, "balconyRateError"],
    [windowCols, "windowColsError"],
    [wallModule, "wallModuleError"],
    [balconyModule, "balconyModuleError"]
  ].forEach(([input, id]) => setFieldError(input, "", id));

  houseFormError?.classList.add("hidden");
  if (houseFormError) houseFormError.textContent = "";
}

function validateHouseForm(showErrors = true) {
  clearHouseValidation();

  const errors = [];
  const floors = parseInt(floorsInput.value, 10);
  const sections = parseInt(sectionsInput.value, 10);
  const width = parseInt(widthInput.value, 10);
  const depth = parseInt(depthInput.value, 10);
  const balconyRateValueNum = parseFloat(balconyRate.value);
  const windowColsNum = parseInt(windowCols.value, 10);

  const maxReasonableSections = Math.max(1, Math.floor(width / 2));

  if (!wallModule.value) {
    errors.push("Wall module is required.");
    if (showErrors) setFieldError(wallModule, "Select a wall module.", "wallModuleError");
  }

  if (!(floors >= 1)) {
    errors.push("Floors must be at least 1.");
    if (showErrors) setFieldError(floorsInput, "Floors must be at least 1.", "floorsError");
  }

  if (!(width >= 6)) {
    errors.push("Width must be at least 6.");
    if (showErrors) setFieldError(widthInput, "Width must be at least 6.", "widthError");
  }

  if (!(depth >= 1)) {
    errors.push("Depth must be at least 1.");
    if (showErrors) setFieldError(depthInput, "Depth must be at least 1.", "depthError");
  }

  if (sections > maxReasonableSections) {
    errors.push("Too many sections for the selected width.");
    if (showErrors) setFieldError(sectionsInput, `Use ${maxReasonableSections} sections or fewer for this width.`, "sectionsError");
  }

  if (windowColsNum > width) {
    errors.push("Window columns cannot exceed house width.");
    if (showErrors) setFieldError(windowCols, "Window columns cannot exceed width.", "windowColsError");
  }

  if (hasBalconies.checked && !balconyModule.value) {
    errors.push("Balcony module is required when balconies are enabled.");
    if (showErrors) setFieldError(balconyModule, "Select a balcony module or disable balconies.", "balconyModuleError");
  }

  if (balconyRateValueNum < 0 || balconyRateValueNum > 1) {
    errors.push("Balcony rate must be between 0 and 1.");
    if (showErrors) setFieldError(balconyRate, "Balcony rate must be between 0 and 1.", "balconyRateError");
  }

  if (showErrors && errors.length) {
    houseFormError.textContent = "Please fix the highlighted house fields.";
    houseFormError.classList.remove("hidden");
  }

  return { valid: errors.length === 0, errors };
}

// ================= TEXTURE HELPERS =================
function getTextureAbsoluteUrl(url) {
  if (!url) return "";
  if (/^https?:\/\//i.test(url)) return url;
  if (url.startsWith("/")) return SERVER_URL + url;
  return `${SERVER_URL}/${url.replace(/^\/+/, "")}`;
}

function getTextureFromCache(url) {
  const THREE = window.THREE;
  if (!url) return null;

  const absoluteUrl = getTextureAbsoluteUrl(url);
  if (textureCache.has(absoluteUrl)) return textureCache.get(absoluteUrl);

  const texture = new THREE.TextureLoader().load(absoluteUrl);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.userData = { sharedTexture: true };
  textureCache.set(absoluteUrl, texture);
  return texture;
}

// ================= FACADE / PREFAB HELPERS =================
function getWallBaseColor(house, wallData) {
  return wallData?.params?.color || DEFAULTS.colors.wall;
}

function getFacadeSettings(house) {
  return {
    texture_url: house?.facade?.texture_url || "",
    texture_scale: house?.facade?.texture_scale ?? parseFloat(facadeTextureRepeat?.value || DEFAULTS.facade.textureScale)
  };
}

function buildFacadeMaterial(THREE, tintColor, width, floors, textureUrl, textureScale = DEFAULTS.facade.textureScale) {
  const color = new THREE.Color(tintColor || DEFAULTS.colors.wall);

  if (!textureUrl) {
    const material = new THREE.MeshStandardMaterial({ color, roughness: 0.88, metalness: 0.05 });
    material.userData.shared = false;
    material.userData.disposed = false;
    return material;
  }

  const texture = getTextureFromCache(textureUrl);
  if (!texture) {
    const material = new THREE.MeshStandardMaterial({ color, roughness: 0.88, metalness: 0.05 });
    material.userData.shared = false;
    material.userData.disposed = false;
    return material;
  }

  texture.repeat.set(
    Math.max(1, width / Math.max(1, textureScale)),
    Math.max(1, floors / Math.max(1, textureScale))
  );

  const material = new THREE.MeshStandardMaterial({
    color,
    map: texture,
    roughness: 0.9,
    metalness: 0.03
  });
  material.userData.shared = false;
  material.userData.disposed = false;
  return material;
}

function getSharedMaterials() {
  const THREE = window.THREE;

  const entranceFrame = getSharedMaterial("shared_entrance_frame", () => {
    return new THREE.MeshStandardMaterial({ color: DEFAULTS.colors.entranceFrame, roughness: 0.7 });
  });

  const seam = getSharedMaterial("shared_seam", () => {
    return new THREE.MeshStandardMaterial({ color: DEFAULTS.colors.seam, roughness: 0.85 });
  });

  return { entranceFrame, seam };
}

function normalizeModuleForBuilding(type, mod) {
  if (!mod || !mod.params) return mod;

  const clone = JSON.parse(JSON.stringify(mod));

  if (type === "wall") {
    clone.params.width = clamp(clone.params.width || 0.95, DEFAULTS.normalize.wall.widthMin, DEFAULTS.normalize.wall.widthMax);
    clone.params.height = clamp(clone.params.height || 1.1, DEFAULTS.normalize.wall.heightMin, DEFAULTS.normalize.wall.heightMax);
    clone.params.depth = clamp(clone.params.depth || 0.06, DEFAULTS.normalize.wall.depthMin, DEFAULTS.normalize.wall.depthMax);
  }

  if (type === "window") {
    clone.params.width = clamp(clone.params.width || 0.42, DEFAULTS.normalize.window.widthMin, DEFAULTS.normalize.window.widthMax);
    clone.params.height = clamp(clone.params.height || 0.42, DEFAULTS.normalize.window.heightMin, DEFAULTS.normalize.window.heightMax);
    clone.params.depth = DEFAULTS.normalize.window.depth;
  }

  if (type === "door") {
    clone.params.width = clamp(clone.params.width || 0.55, DEFAULTS.normalize.door.widthMin, DEFAULTS.normalize.door.widthMax);
    clone.params.height = clamp(clone.params.height || 0.9, DEFAULTS.normalize.door.heightMin, DEFAULTS.normalize.door.heightMax);
    clone.params.depth = DEFAULTS.normalize.door.depth;
  }

  if (type === "balcony") {
    clone.params.width = clamp(clone.params.width || 0.72, DEFAULTS.normalize.balcony.widthMin, DEFAULTS.normalize.balcony.widthMax);
    clone.params.depth = clamp(clone.params.depth || 0.5, DEFAULTS.normalize.balcony.depthMin, DEFAULTS.normalize.balcony.depthMax);
  }

  return clone;
}

function resolveCompositeModule(wallData, windowData, doorData, balconyData, roofData) {
  const wallParams = wallData?.params || {};
  const roofParams = roofData?.params || {};

  return {
    wall: wallData,
    roof: roofData,
    width: wallParams.width || 0.95,
    height: wallParams.height || 1.1,
    depth: wallParams.depth || 0.06,
    color: wallParams.color || DEFAULTS.colors.wall,

    hasWindow: Boolean(wallParams.has_window || windowData?.params),
    hasDoor: Boolean(wallParams.has_door || doorData?.params),
    hasBalcony: Boolean(wallParams.has_balcony || balconyData?.params),

    windowSource: wallParams.has_window ? wallData : windowData,
    doorSource: wallParams.has_door ? wallData : doorData,
    balconySource: wallParams.has_balcony ? wallData : balconyData,
    roofSource: roofData || null,

    roofHeight: roofParams.height || DEFAULTS.module.roof.height,
    roofDepth: roofParams.depth || DEFAULTS.module.roof.depth,
    roofWidth: roofParams.width || DEFAULTS.module.roof.width,
    roofColor: roofParams.color || DEFAULTS.colors.roof
  };
}

function getFeatureDims(source, defaults) {
  const p = source?.params || {};
  return {
    width: p.width ?? defaults.width,
    height: p.height ?? defaults.height,
    depth: p.depth ?? defaults.depth,
    color: p.color ?? defaults.color
  };
}

function buildFacadeVariationPattern(width, sections) {
  const sectionBlindChance = [];
  for (let i = 0; i < sections; i++) {
    sectionBlindChance.push(Math.random() * 0.08);
  }

  return {
    sectionBlindChance,
    frontShift: randInt(0, DEFAULTS.facade.sectionShiftMax),
    backShift: randInt(0, DEFAULTS.facade.sectionShiftMax)
  };
}

function getEvenlySpacedColumns(totalCols, targetCount, blockedCols = [], shift = 0) {
  const blocked = new Set(blockedCols);
  const available = [];

  for (let col = 0; col < totalCols; col++) {
    if (!blocked.has(col)) available.push(col);
  }

  if (!available.length) return [];

  const count = Math.min(targetCount, available.length);
  if (count <= 0) return [];
  if (count === available.length) return [...available];

  const result = [];
  const step = (available.length - 1) / Math.max(1, count - 1);

  for (let i = 0; i < count; i++) {
    const index = Math.round(i * step);
    const base = available[Math.min(index, available.length - 1)];
    const shifted = (base + shift) % totalCols;
    if (!blocked.has(shifted)) result.push(shifted);
    else result.push(base);
  }

  return [...new Set(result)].sort((a, b) => a - b);
}

function getRandomSubset(items, rate = 0.25, maxRate = DEFAULTS.facade.balconyMaxRate) {
  if (!Array.isArray(items) || !items.length || rate <= 0) return [];

  const clampedRate = Math.max(0, Math.min(maxRate, rate));
  const pool = [...items];
  const result = [];
  const targetCount = Math.floor(pool.length * clampedRate);
  if (targetCount <= 0) return [];

  while (result.length < targetCount && pool.length) {
    const index = Math.floor(Math.random() * pool.length);
    result.push(pool[index]);
    pool.splice(index, 1);
  }

  return result.sort((a, b) => a - b);
}

function getSectionIndex(col, width, sections) {
  const sectionWidth = width / sections;
  return Math.min(sections - 1, Math.max(0, Math.floor(col / sectionWidth)));
}

function resolveCellContext(side, floor, col, house, entranceCols, sectionIndex) {
  const isGround = floor === 0;
  const isEntranceZone = side === "front" && isGround && entranceCols.includes(col);
  const isFront = side === "front";
  const isBack = side === "back";
  const isTopFloor = floor === house.floors - 1;
  const isEdge = col === 0 || col === house.width - 1;

  return {
    side,
    floor,
    col,
    sectionIndex,
    isGround,
    isTopFloor,
    isEdge,
    isEntranceZone,
    allowWindow: isFront || isBack,
    allowDoor: isEntranceZone,
    allowBalcony: (isFront || isBack) && !isGround
  };
}

function resolveCellComposition(cellContext, prefab, house, balconyColsForFloor, windowColsForFloor, facadePattern) {
  const wantsWindow = windowColsForFloor.includes(cellContext.col);
  const wantsBalcony = house.has_balconies && balconyColsForFloor.includes(cellContext.col);

  const sectionBlindChance = facadePattern.sectionBlindChance[cellContext.sectionIndex] || 0;
  const topBoost = cellContext.isTopFloor ? DEFAULTS.facade.topFloorBlindBoost : 0;
  const edgePenalty = cellContext.isEdge ? DEFAULTS.facade.edgeReductionChance : 0;

  const randomBlind = Math.random() < (DEFAULTS.facade.blindPanelChance + sectionBlindChance + topBoost + edgePenalty);

  const allowWindow = cellContext.allowWindow && !cellContext.isEntranceZone && !randomBlind;
  const allowDoor = cellContext.allowDoor;
  const allowBalcony = cellContext.allowBalcony && wantsWindow && !randomBlind;

  return {
    useWindow: allowWindow && wantsWindow && prefab.hasWindow,
    useDoor: allowDoor && prefab.hasDoor,
    useBalcony: allowBalcony && wantsBalcony && prefab.hasBalcony
  };
}

function registerInteractivePanel(panel, THREE, color, label) {
  panel.userData.baseColor = new THREE.Color(color);
  panel.userData.label = label;
  interactiveObjects.push(panel);
}

function addWindowMesh(group, x, y, z, width, height, depth, material) {
  const win = createBox(width, height, depth, material);
  win.position.set(x, y, z);
  group.add(win);
}

function addDoorMesh(group, x, y, z, width, height, depth, material) {
  const door = createBox(width, height, depth, material);
  door.position.set(x, y, z);
  group.add(door);
  return door;
}

function addBalconyMeshes(group, x, y, zPlate, zRailing, width, height, depth, material) {
  const plate = createBox(width, height, depth, material);
  plate.position.set(x, y, zPlate);
  group.add(plate);

  const railing = createBox(width, 0.28, 0.03, material);
  railing.position.set(x, y + 0.12, zRailing);
  group.add(railing);
}

// ================= API / DATA =================
async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  let data = null;
  try {
    data = await response.json();
  } catch {}

  if (!response.ok) {
    const message = data?.detail || data?.error || `Request failed: ${response.status}`;
    throw new Error(message);
  }

  return data;
}

async function loadFacadeTextures() {
  if (!facadeTextureSelect) return;

  try {
    const data = await fetchJson(`${SERVER_URL}/api/facade-textures`);
    facadeTextureSelect.innerHTML = `<option value="">No texture</option>`;

    const list = Array.isArray(data) ? data : [];
    list.forEach(item => {
      const opt = document.createElement("option");

      if (typeof item === "string") {
        opt.value = item;
        opt.textContent = item.split("/").pop();
      } else {
        opt.value = item.url || "";
        opt.textContent = item.name || item.url || "Texture";
      }

      facadeTextureSelect.appendChild(opt);
    });
  } catch {
    facadeTextureSelect.innerHTML = `<option value="">No texture</option>`;
  }
}

function getModuleById(id) {
  return savedModulesData.find(m => m.id === id) || null;
}

function getCurrentHouseProject() {
  return {
    id: `house_${Date.now()}`,
    name: houseName?.value?.trim() || `House ${savedHousesData.length + 1}`,
    created_at: new Date().toISOString(),
    data: getHouseFormData()
  };
}

async function loadSavedModules() {
  try {
    const data = await fetchJson(`${SERVER_URL}/api/modules`);
    savedModulesData = Array.isArray(data) ? data : [];
  } catch {
    savedModulesData = [];
  }

  renderSavedModules();
  populateModuleSelectors();
}

// ================= LIBRARY RENDER =================
function renderEmptyState(container, title, text) {
  container.innerHTML = `
    <div class="empty-state">
      <div class="empty-title">${title}</div>
      <div class="empty-text">${text}</div>
    </div>
  `;
}

function renderSavedModules() {
  if (!savedModules) return;

  savedModules.innerHTML = "";

  const filterValue = libraryFilter?.value || "all";
  const sortValue = librarySort?.value || "date_desc";

  let filtered = filterValue === "all"
    ? [...savedModulesData]
    : savedModulesData.filter(m => m.type === filterValue);

  filtered = sortItems(filtered, sortValue);

  if (!filtered.length) {
    renderEmptyState(
      savedModules,
      "No saved modules",
      "Create and save your first wall, window, door, roof, or balcony module."
    );
    return;
  }

  const fragment = document.createDocumentFragment();

  filtered.forEach(mod => {
    const card = document.createElement("div");
    card.className = "module-card";
    card.dataset.id = mod.id;
    card.innerHTML = `
      <div class="module-card-head">
        <div>
          <div class="module-card-title">${mod.name}</div>
          <div class="module-card-meta">
            Type: ${mod.type} · ${mod.created_at ? new Date(mod.created_at).toLocaleString() : "No date"}
          </div>
        </div>
      </div>
      <div class="module-actions">
        <button data-id="${mod.id}" class="small-btn use-module-btn">Edit</button>
        <button data-id="${mod.id}" class="small-btn rename-module-btn">Rename</button>
        <button data-id="${mod.id}" class="small-btn use-in-builder-btn">Use in Builder</button>
        <button data-id="${mod.id}" class="small-btn delete-module-btn">Delete</button>
      </div>
      <div class="rename-row hidden" id="rename_module_${mod.id}">
        <input class="text-input rename-input" type="text" value="${String(mod.name).replace(/"/g, "&quot;")}" />
        <button data-id="${mod.id}" class="small-btn save-rename-module-btn">Save</button>
        <button data-id="${mod.id}" class="small-btn cancel-rename-module-btn">Cancel</button>
      </div>
    `;
    fragment.appendChild(card);
  });

  savedModules.appendChild(fragment);

  savedModules.querySelectorAll(".use-module-btn").forEach(btn => {
    btn.onclick = () => {
      const mod = savedModulesData.find(m => m.id === btn.dataset.id);
      if (!mod) return;
      applyModuleParams(mod);
      renderModulePreview(mod);
      setActiveTab("modulePage");
    };
  });

  savedModules.querySelectorAll(".rename-module-btn").forEach(btn => {
    btn.onclick = () => $(`rename_module_${btn.dataset.id}`)?.classList.remove("hidden");
  });

  savedModules.querySelectorAll(".cancel-rename-module-btn").forEach(btn => {
    btn.onclick = () => $(`rename_module_${btn.dataset.id}`)?.classList.add("hidden");
  });

  savedModules.querySelectorAll(".save-rename-module-btn").forEach(btn => {
    btn.onclick = async () => {
      const row = $(`rename_module_${btn.dataset.id}`);
      const input = row?.querySelector(".rename-input");
      const newName = input?.value?.trim();

      if (!newName) {
        showToast("Module name cannot be empty.", "error");
        return;
      }

      const mod = savedModulesData.find(m => m.id === btn.dataset.id);
      if (!mod) return;

      mod.name = newName;

      try {
        await fetchJson(`${SERVER_URL}/api/modules/${mod.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: newName })
        });
      } catch {}

      renderSavedModules();
      populateModuleSelectors();
      showToast("Module renamed successfully.", "success");
    };
  });

  savedModules.querySelectorAll(".use-in-builder-btn").forEach(btn => {
    btn.onclick = () => {
      const mod = savedModulesData.find(m => m.id === btn.dataset.id);
      if (!mod) return;

      if (mod.type === "wall") wallModule.value = mod.id;
      if (mod.type === "window") windowModule.value = mod.id;
      if (mod.type === "door") doorModule.value = mod.id;
      if (mod.type === "roof") roofModule.value = mod.id;
      if (mod.type === "balcony") balconyModule.value = mod.id;

      updateHouseJsonPreview();
      setActiveTab("builder");
      showToast(`Module "${mod.name}" assigned to builder.`, "success");
    };
  });

  savedModules.querySelectorAll(".delete-module-btn").forEach(btn => {
    btn.onclick = async () => {
      const id = btn.dataset.id;

      try {
        await fetchJson(`${SERVER_URL}/api/modules/${id}`, { method: "DELETE" });
      } catch {}

      savedModulesData = savedModulesData.filter(m => m.id !== id);
      renderSavedModules();
      populateModuleSelectors();
      showToast("Module deleted.", "success");
    };
  });
}

function renderSavedHouses() {
  if (!savedHouses) return;

  savedHouses.innerHTML = "";

  let items = sortItems(savedHousesData, houseSort?.value || "date_desc");

  if (!items.length) {
    renderEmptyState(
      savedHouses,
      "No saved houses",
      "Build a house, adjust the facade, and save it here for later comparison."
    );
    return;
  }

  const fragment = document.createDocumentFragment();

  items.forEach(item => {
    const card = document.createElement("div");
    card.className = "module-card";
    card.dataset.id = item.id;
    card.innerHTML = `
      <div class="module-card-head">
        <div>
          <div class="module-card-title">${item.name || "Unnamed House"}</div>
          <div class="module-card-meta">${item.created_at ? new Date(item.created_at).toLocaleString() : ""}</div>
        </div>
      </div>
      <div class="module-actions">
        <button class="small-btn load-house-btn" data-id="${item.id}">Load</button>
        <button class="small-btn rename-house-btn" data-id="${item.id}">Rename</button>
        <button class="small-btn delete-house-btn" data-id="${item.id}">Delete</button>
      </div>
      <div class="rename-row hidden" id="rename_house_${item.id}">
        <input class="text-input rename-house-input" type="text" value="${String(item.name || "").replace(/"/g, "&quot;")}" />
        <button class="small-btn save-rename-house-btn" data-id="${item.id}">Save</button>
        <button class="small-btn cancel-rename-house-btn" data-id="${item.id}">Cancel</button>
      </div>
    `;
    fragment.appendChild(card);
  });

  savedHouses.appendChild(fragment);

  savedHouses.querySelectorAll(".load-house-btn").forEach(btn => {
    btn.onclick = () => {
      const item = savedHousesData.find(h => h.id === btn.dataset.id);
      if (!item) return;

      if (houseName) houseName.value = item.name || "";
      applyHouseParams(item.data);
      generateHousePreview();
      showToast(`House "${item.name}" loaded.`, "success");
    };
  });

  savedHouses.querySelectorAll(".rename-house-btn").forEach(btn => {
    btn.onclick = () => $(`rename_house_${btn.dataset.id}`)?.classList.remove("hidden");
  });

  savedHouses.querySelectorAll(".cancel-rename-house-btn").forEach(btn => {
    btn.onclick = () => $(`rename_house_${btn.dataset.id}`)?.classList.add("hidden");
  });

  savedHouses.querySelectorAll(".save-rename-house-btn").forEach(btn => {
    btn.onclick = () => {
      const item = savedHousesData.find(h => h.id === btn.dataset.id);
      const row = $(`rename_house_${btn.dataset.id}`);
      const input = row?.querySelector(".rename-house-input");
      const newName = input?.value?.trim();

      if (!item || !newName) {
        showToast("House name cannot be empty.", "error");
        return;
      }

      item.name = newName;
      renderSavedHouses();
      showToast("House renamed successfully.", "success");
    };
  });

  savedHouses.querySelectorAll(".delete-house-btn").forEach(btn => {
    btn.onclick = () => {
      savedHousesData = savedHousesData.filter(h => h.id !== btn.dataset.id);
      renderSavedHouses();
      showToast("House deleted.", "success");
    };
  });
}

function populateSelectWithModules(select, modules) {
  if (!select) return;

  const currentValue = select.value;
  select.innerHTML = "";

  if (!modules.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No modules";
    select.appendChild(opt);
    return;
  }

  const fragment = document.createDocumentFragment();

  sortItems(modules, "name_asc").forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.name;
    fragment.appendChild(opt);
  });

  select.appendChild(fragment);

  if ([...select.options].some(opt => opt.value === currentValue)) {
    select.value = currentValue;
  }
}

function populateModuleSelectors() {
  const byType = type => savedModulesData.filter(m => m.type === type);

  populateSelectWithModules(wallModule, byType("wall"));
  populateSelectWithModules(windowModule, byType("window"));
  populateSelectWithModules(doorModule, byType("door"));
  populateSelectWithModules(roofModule, byType("roof"));
  populateSelectWithModules(balconyModule, byType("balcony"));
}

// ================= MODULE PREVIEW =================
function renderModulePreview(data) {
  clearSceneMeshes();

  const THREE = window.THREE;
  const group = new THREE.Group();
  const p = data.params || data;

  const width = p.width || DEFAULTS.module.wall.width;
  const height = p.height || DEFAULTS.module.wall.height;
  const depth = p.depth || DEFAULTS.module.wall.depth;
  const color = p.color || DEFAULTS.colors.wall;

  const material = new THREE.MeshStandardMaterial({
    color: color,
    roughness: 0.8,
    metalness: 0.1
  });

  const moduleColor = color || DEFAULTS.colors.wall;

  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: moduleColor,
    roughness: 0.9,
    metalness: 0.03
  });

  const windowMaterial = new THREE.MeshStandardMaterial({
    color: DEFAULTS.colors.window,
    roughness: 0.5,
    metalness: 0.08
  });

  const doorMaterial = new THREE.MeshStandardMaterial({
    color: DEFAULTS.colors.door,
    roughness: 0.8
  });

  const balconyMaterial = new THREE.MeshStandardMaterial({
    color: DEFAULTS.colors.balcony,
    roughness: 0.75
  });

  const roofMaterial = new THREE.MeshStandardMaterial({
    color: DEFAULTS.colors.roof,
    roughness: 0.85
  });

  const body = createBox(width, height, depth, material);
  body.position.y = height / 2;
  group.add(body);

  if (p.has_window) {
    const win = createBox(width * 0.5, height * 0.4, 0.03, windowMaterial);
    win.position.set(0, height * 0.55, depth / 2 + 0.02);
    group.add(win);
  }

  if (p.has_door) {
    const door = createBox(width * 0.35, height * 0.55, 0.03, doorMaterial);
    door.position.set(0, height * 0.28, depth / 2 + 0.02);
    group.add(door);
  }

  if (p.has_balcony) {
    const balcony = createBox(width * 0.7, 0.12, Math.max(0.22, depth * 1.5), balconyMaterial);
    balcony.position.set(0, height * 0.35, depth / 2 + Math.max(0.18, depth * 0.9));
    group.add(balcony);
  }

  if (p.is_roof_piece) {
    const roof = createBox(width + 0.15, Math.max(0.08, height * 0.4), depth + 0.15, roofMaterial);
    roof.position.set(0, height + Math.max(0.04, height * 0.2), 0);
    group.add(roof);
  }

  currentMesh = group;
  scene.add(group);
  frameObject(currentMesh);
}

function liveUpdateModulePreview() {
  if (modulePage?.classList.contains("hidden")) return;
  updateModuleJsonPreview();
  renderModulePreview(getModuleFormData());
}

// ================= HOUSE RENDER =================
function renderHousePreview(payload, offsetX = 0) {
  const THREE = window.THREE;
  const { entranceFrame, seam } = getSharedMaterials();

  const group = new THREE.Group();
  const house = { ...payload.house };

  house.floors = clamp(house.floors || 9, DEFAULTS.house.minFloors, DEFAULTS.house.maxFloors);
  house.width = clamp(house.width || 16, DEFAULTS.house.minWidth, DEFAULTS.house.maxWidth);
  house.depth = clamp(house.depth || 2, DEFAULTS.house.minDepth, DEFAULTS.house.maxDepth);
  house.sections = clamp(
    house.sections || 3,
    DEFAULTS.house.minSections,
    Math.min(DEFAULTS.house.maxSections, Math.max(1, Math.floor(house.width / 2)))
  );
  house.window_cols = clamp(house.window_cols || 8, DEFAULTS.house.minWindowCols, house.width);
  house.balcony_rate = clamp(house.balcony_rate || 0.25, 0, 1);
  house.roof_type = DEFAULTS.house.roofType;

  const floors = house.floors;
  const sections = house.sections;
  const width = house.width;
  const depth = house.depth;
  const floorHeight = DEFAULTS.house.floorHeight;
  const buildingHeight = floors * floorHeight;

  const wallData = normalizeModuleForBuilding("wall", getModuleById(payload.modules.wall));
  const windowData = normalizeModuleForBuilding("window", getModuleById(payload.modules.window));
  const doorData = normalizeModuleForBuilding("door", getModuleById(payload.modules.door));
  const roofData = normalizeModuleForBuilding("roof", getModuleById(payload.modules.roof));
  const balconyData = normalizeModuleForBuilding("balcony", getModuleById(payload.modules.balcony));

  const prefab = resolveCompositeModule(wallData, windowData, doorData, balconyData, roofData);
  const wallColor = getWallBaseColor(house, wallData);
  const facade = getFacadeSettings(house);
  const facadePattern = buildFacadeVariationPattern(width, sections);

  const panelWidth = prefab.width;
  const panelHeight = Math.min(Math.max(prefab.height, DEFAULTS.normalize.wall.heightMin), DEFAULTS.normalize.wall.heightMax);
  const panelDepth = Math.min(Math.max(prefab.depth, DEFAULTS.normalize.wall.depthMin), DEFAULTS.normalize.wall.depthMax);

  const windowDims = getFeatureDims(prefab.windowSource, {
    width: 0.42,
    height: 0.42,
    depth: 0.05,
    color: DEFAULTS.colors.window
  });

  const doorDims = getFeatureDims(prefab.doorSource, {
    width: 0.55,
    height: 0.9,
    depth: 0.05,
    color: DEFAULTS.colors.door
  });

  const balconyDims = getFeatureDims(prefab.balconySource, {
    width: 0.72,
    height: 0.18,
    depth: 0.5,
    color: DEFAULTS.colors.balcony
  });

  const roofOverhang = Math.max(DEFAULTS.roof.flatOverhang, (prefab.roofWidth - 1) * 0.2 + DEFAULTS.roof.flatOverhang);
  const roofThickness = Math.max(DEFAULTS.roof.minThickness, prefab.roofHeight);
  const roofProjection = Math.max(DEFAULTS.roof.flatOverhang, prefab.roofDepth);

  const bodyMaterial = createMaterialVariant(
    getSharedMaterial(`body_${wallColor}`, () => {
      return new THREE.MeshStandardMaterial({ color: wallColor, roughness: 0.92, metalness: 0.02 });
    })
  );

  const body = createBox(
    Math.max(0.2, width - panelDepth * 1.2),
    buildingHeight,
    Math.max(0.2, depth - panelDepth * 1.2),
    bodyMaterial
  );
  body.position.set(offsetX, buildingHeight / 2, 0);
  group.add(body);

  const panelMaterialFrontBack = buildFacadeMaterial(
    THREE,
    wallColor,
    width,
    floors,
    facade.texture_url,
    facade.texture_scale
  );

  const panelMaterialSide = buildFacadeMaterial(
    THREE,
    wallColor,
    depth,
    floors,
    facade.texture_url,
    facade.texture_scale
  );

  const windowMaterial = createMaterialVariant(
    getSharedMaterial("window_default", () => {
      return new THREE.MeshStandardMaterial({ color: DEFAULTS.colors.window, roughness: 0.45, metalness: 0.1 });
    }),
    windowDims.color
  );

  const doorMaterial = createMaterialVariant(
    getSharedMaterial("door_default", () => {
      return new THREE.MeshStandardMaterial({ color: DEFAULTS.colors.door, roughness: 0.78 });
    }),
    doorDims.color
  );

  const balconyMaterial = createMaterialVariant(
    getSharedMaterial("balcony_default", () => {
      return new THREE.MeshStandardMaterial({ color: DEFAULTS.colors.balcony, roughness: 0.8 });
    }),
    balconyDims.color
  );

  const roofMaterial = createMaterialVariant(
    getSharedMaterial("roof_default", () => {
      return new THREE.MeshStandardMaterial({ color: DEFAULTS.colors.roof, roughness: 0.86 });
    }),
    prefab.roofColor
  );

  const entranceCols = [];
  for (let s = 0; s < sections; s++) {
    const centerCol = Math.round(((s + 0.5) * width) / sections - 0.5);
    entranceCols.push(Math.max(0, Math.min(width - 1, centerCol)));
  }

  const frontWindowColsGround = getEvenlySpacedColumns(width, house.window_cols, entranceCols, facadePattern.frontShift);
  const frontWindowColsUpper = getEvenlySpacedColumns(width, house.window_cols, [], facadePattern.frontShift);
  const backWindowColsAll = getEvenlySpacedColumns(width, house.window_cols, [], facadePattern.backShift);

  const frontWindowsByFloor = [];
  const backWindowsByFloor = [];
  const frontBalconiesByFloor = [];
  const backBalconiesByFloor = [];

  for (let floor = 0; floor < floors; floor++) {
    frontWindowsByFloor[floor] = floor === 0 ? [...frontWindowColsGround] : [...frontWindowColsUpper];
    backWindowsByFloor[floor] = [...backWindowColsAll];

    let frontRate = house.balcony_rate;
    let backRate = Math.max(0, house.balcony_rate - 0.05);

    if (floor === floors - 1) {
      frontRate = Math.max(0, frontRate - 0.08);
      backRate = Math.max(0, backRate - 0.08);
    }

    frontBalconiesByFloor[floor] = floor === 0 ? [] : getRandomSubset(frontWindowsByFloor[floor], frontRate, DEFAULTS.facade.balconyMaxRate);
    backBalconiesByFloor[floor] = floor === 0 ? [] : getRandomSubset(backWindowsByFloor[floor], backRate, DEFAULTS.facade.balconyMaxRate);
  }

  function renderResolvedCell(cellContext, composition, pos) {
    const panel = createBox(panelWidth, panelHeight, panelDepth, pos.panelMaterial);
    panel.position.set(pos.x, pos.yCenter, pos.zPanel);
    registerInteractivePanel(
      panel,
      THREE,
      wallColor,
      `${cellContext.side} • Floor ${cellContext.floor + 1}, Slot ${cellContext.col + 1}`
    );
    group.add(panel);

    if (composition.useWindow) {
      addWindowMesh(group, pos.x, pos.yCenter, pos.zWindow, windowDims.width, windowDims.height, 0.05, windowMaterial);
    }

    if (composition.useDoor) {
      const frame = createBox(1.05, 1.25, 0.03, entranceFrame);
      frame.position.set(pos.x, 0.62, pos.zDoor - Math.sign(pos.zDoor) * 0.03);
      group.add(frame);

      const door = addDoorMesh(group, pos.x, 0.45, pos.zDoor, doorDims.width, doorDims.height, 0.05, doorMaterial);
      door.userData.baseColor = new THREE.Color(doorDims.color);
      door.userData.label = `Entrance ${entranceCols.indexOf(cellContext.col) + 1}`;
      interactiveObjects.push(door);
    }

    if (composition.useBalcony) {
      addBalconyMeshes(
        group,
        pos.x,
        cellContext.floor * floorHeight + 0.2,
        pos.zBalcony,
        pos.zRailing,
        balconyDims.width,
        0.18,
        balconyDims.depth,
        balconyMaterial
      );
    }
  }

  function renderFrontBack(sideSign, sideName, windowsByFloor, balconiesByFloor, withEntrances) {
    const baseZ = sideSign === 1 ? depth / 2 : -depth / 2;
    const panelOffset = sideSign * (panelDepth / 2 + 0.002);
    const featureBaseZ = sideSign * (Math.abs(baseZ) + panelDepth);

    const zPanel = baseZ + panelOffset;
    const zWindow = featureBaseZ + sideSign * (0.05 / 2 + 0.01);
    const zDoor = featureBaseZ + sideSign * (0.05 / 2 + 0.01);
    const zBalcony = featureBaseZ + sideSign * (0.05 + balconyDims.depth / 2 + 0.04);
    const zRailing = featureBaseZ + sideSign * (0.05 + balconyDims.depth + 0.03);

    for (let floor = 0; floor < floors; floor++) {
      for (let col = 0; col < width; col++) {
        const sectionIndex = getSectionIndex(col, width, sections);
        const cellContext = resolveCellContext(sideName, floor, col, house, entranceCols, sectionIndex);
        const x = offsetX + col - width / 2 + 0.5;
        const yCenter = floor * floorHeight + floorHeight / 2;

        const composition = resolveCellComposition(
          cellContext,
          prefab,
          house,
          balconiesByFloor[floor],
          windowsByFloor[floor],
          facadePattern
        );

        renderResolvedCell(cellContext, composition, {
          x,
          yCenter,
          zPanel,
          zWindow,
          zDoor,
          zBalcony,
          zRailing,
          panelMaterial: panelMaterialFrontBack
        });
      }
    }

    if (withEntrances) {
      for (let s = 1; s < sections; s++) {
        const seamX = offsetX - width / 2 + (s * width / sections);
        const seamPart = createBox(0.08, buildingHeight, 0.08, seam);
        seamPart.position.set(seamX, buildingHeight / 2, sideSign * (depth / 2 + 0.05));
        group.add(seamPart);
      }
    }
  }

  function renderSideWalls() {
    const sidePanelDepth = panelDepth;
    const rightX = offsetX + width / 2 + sidePanelDepth / 2 + 0.002;
    const leftX = offsetX - width / 2 - sidePanelDepth / 2 - 0.002;

    for (let floor = 0; floor < floors; floor++) {
      const yCenter = floor * floorHeight + floorHeight / 2;

      for (let d = 0; d < depth; d++) {
        const z = d - depth / 2 + 0.5;

        const rightPanel = createBox(sidePanelDepth, panelHeight, 1, panelMaterialSide);
        rightPanel.position.set(rightX, yCenter, z);
        registerInteractivePanel(rightPanel, THREE, wallColor, `right • Floor ${floor + 1}, Segment ${d + 1}`);
        group.add(rightPanel);

        const leftPanel = createBox(sidePanelDepth, panelHeight, 1, panelMaterialSide);
        leftPanel.position.set(leftX, yCenter, z);
        registerInteractivePanel(leftPanel, THREE, wallColor, `left • Floor ${floor + 1}, Segment ${d + 1}`);
        group.add(leftPanel);
      }
    }
  }

  renderFrontBack(1, "front", frontWindowsByFloor, frontBalconiesByFloor, true);
  renderFrontBack(-1, "back", backWindowsByFloor, backBalconiesByFloor, false);
  renderSideWalls();

  const roof = createBox(
    width + roofOverhang,
    roofThickness,
    depth + roofProjection,
    roofMaterial
  );
  roof.position.set(offsetX, buildingHeight + roofThickness / 2, 0);
  group.add(roof);

  const parapet = createBox(
    width + roofOverhang * 0.9,
    roofThickness * 0.35,
    depth + roofProjection * 0.9,
    roofMaterial
  );
  parapet.position.set(offsetX, buildingHeight + roofThickness + roofThickness * 0.18, 0);
  group.add(parapet);

  return group;
}

function generateHousePreview() {
  const validation = validateHouseForm(true);
  if (!validation.valid) {
    showToast("Please fix house form errors before generating.", "error");
    return;
  }

  clearSceneMeshes();

  const payload = getHouseFormData();
  currentMesh = renderHousePreview(payload, 0);
  scene.add(currentMesh);

  updateHouseJsonPreview();
  frameObject(currentMesh);
}

async function animateHousePreview() {
  const validation = validateHouseForm(true);
  if (!validation.valid) {
    showToast("Please fix house form errors before animation.", "error");
    return;
  }

  clearSceneMeshes();

  const payload = getHouseFormData();
  const house = payload.house;
  const wallData = normalizeModuleForBuilding("wall", getModuleById(payload.modules.wall));
  const shellColor = getWallBaseColor(house, wallData);

  const group = new window.THREE.Group();
  currentMesh = group;
  scene.add(group);

  const floors = Math.max(DEFAULTS.house.minFloors, house.floors || 9);
  const width = Math.max(DEFAULTS.house.minWidth, house.width || 16);
  const depth = Math.max(DEFAULTS.house.minDepth, house.depth || 2);
  const floorHeight = DEFAULTS.house.floorHeight;
  const buildingHeight = floors * floorHeight;

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  const shellMaterial = createMaterialVariant(
    getSharedMaterial(`anim_shell_${shellColor}`, () => {
      return new window.THREE.MeshStandardMaterial({ color: shellColor, roughness: 0.9 });
    })
  );

  const slabMaterial = getSharedMaterial("anim_slab", () => {
    return new window.THREE.MeshStandardMaterial({ color: 0x999999, roughness: 0.85 });
  });

  const body = createBox(width, buildingHeight, depth, shellMaterial);
  body.position.set(0, buildingHeight / 2, 0);
  group.add(body);
  frameObject(group);
  await sleep(180);

  for (let floor = 0; floor < floors; floor++) {
    const slab = createBox(width + 0.1, 0.03, depth + 0.1, slabMaterial);
    slab.position.set(0, floor * floorHeight + floorHeight, 0);
    group.add(slab);
    await sleep(70);
  }

  clearMesh(group);
  currentMesh = renderHousePreview(payload, 0);
  scene.add(currentMesh);

  updateHouseJsonPreview();
  frameObject(currentMesh);
}

function randomizeHouse() {
  const textureOptions = [...(facadeTextureSelect?.options || [])].map(opt => opt.value).filter(Boolean);

  const width = randInt(DEFAULTS.randomize.width[0], DEFAULTS.randomize.width[1]);
  const maxSections = Math.min(DEFAULTS.randomize.sections[1], Math.max(1, Math.floor(width / 2)));
  const sections = randInt(DEFAULTS.randomize.sections[0], maxSections);

  widthInput.value = width;
  floorsInput.value = randInt(DEFAULTS.randomize.floors[0], DEFAULTS.randomize.floors[1]);
  depthInput.value = randInt(DEFAULTS.randomize.depth[0], DEFAULTS.randomize.depth[1]);
  sectionsInput.value = sections;
  windowCols.value = clamp(
    randInt(DEFAULTS.randomize.windowCols[0], DEFAULTS.randomize.windowCols[1]),
    DEFAULTS.house.minWindowCols,
    width
  );

  hasBalconies.checked = Math.random() > 0.2;
  balconyRate.value = clamp(randFloat(DEFAULTS.randomize.balconyRate[0], DEFAULTS.randomize.balconyRate[1], 0.05), 0, 1);
  facadeTextureRepeat.value = clamp(randInt(DEFAULTS.randomize.textureScale[0], DEFAULTS.randomize.textureScale[1]), 1, 8);

  if (facadeTextureSelect) {
    facadeTextureSelect.value = textureOptions.length && Math.random() > 0.35
      ? textureOptions[randInt(0, textureOptions.length - 1)]
      : "";
  }

  updateHouseRangeLabels();
  updateHouseJsonPreview();
  generateHousePreview();
  showToast("House randomized.", "success");
}

function randomizeFacade() {
  const textureOptions = [...(facadeTextureSelect?.options || [])].map(opt => opt.value).filter(Boolean);

  hasBalconies.checked = Math.random() > 0.15;
  balconyRate.value = clamp(randFloat(DEFAULTS.randomize.balconyRate[0], DEFAULTS.randomize.balconyRate[1], 0.05), 0, 1);
  windowCols.value = clamp(
    randInt(DEFAULTS.randomize.windowCols[0], DEFAULTS.randomize.windowCols[1]),
    DEFAULTS.house.minWindowCols,
    parseInt(widthInput.value, 10)
  );
  facadeTextureRepeat.value = clamp(randInt(DEFAULTS.randomize.textureScale[0], DEFAULTS.randomize.textureScale[1]), 1, 8);

  if (facadeTextureSelect) {
    facadeTextureSelect.value = textureOptions.length && Math.random() > 0.3
      ? textureOptions[randInt(0, textureOptions.length - 1)]
      : "";
  }

  updateHouseRangeLabels();
  updateHouseJsonPreview();
  generateHousePreview();
  showToast("Facade randomized.", "success");
}

// ================= IMPORT / EXPORT =================
async function exportLibraryZip() {
  const zip = new JSZip();

  if (!savedModulesData.length) throw new Error("Library is empty");

  const payload = {
    type: "module_library",
    version: 2,
    exported_at: new Date().toISOString(),
    modules: [...savedModulesData]
  };

  zip.file("library.json", JSON.stringify(payload, null, 2));

  const originalPreviewJson = previewJson?.textContent || "{}";
  const currentModuleData = getModuleFormData();

  renderModulePreview(savedModulesData[0]);
  await nextFrame();
  zip.file("library-preview.png", getRendererPreviewBase64(), { base64: true });

  for (let i = 0; i < savedModulesData.length; i++) {
    const mod = savedModulesData[i];
    renderModulePreview(mod);
    await nextFrame();

    const fileName = `modules/${String(i + 1).padStart(3, "0")}_${sanitizeFilename(mod.name || mod.id || "module")}.png`;
    zip.file(fileName, getRendererPreviewBase64(), { base64: true });
  }

  applyModuleParams(currentModuleData);
  renderModulePreview(currentModuleData);
  if (previewJson) previewJson.textContent = originalPreviewJson;

  const blob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "module-library.zip";
  a.click();

  URL.revokeObjectURL(url);
}

async function importLibraryZip(file) {
  const zip = await JSZip.loadAsync(file);
  const libraryFile = zip.file("library.json");

  if (!libraryFile) throw new Error("library.json not found in ZIP");

  const text = await libraryFile.async("text");
  const data = JSON.parse(text);

  if (data.type !== "module_library" || !Array.isArray(data.modules)) {
    throw new Error("Invalid library file");
  }

  savedModulesData = data.modules;
  renderSavedModules();
  populateModuleSelectors();

  if (savedModulesData.length) {
    applyModuleParams(savedModulesData[0]);
    renderModulePreview(savedModulesData[0]);
  }

  setActiveTab("library");
}

async function exportProjectZip() {
  const zip = new JSZip();

  const payload = {
    type: "panel_project",
    version: 2,
    exported_at: new Date().toISOString(),
    modules: savedModulesData,
    houses: savedHousesData,
    current_house: getHouseFormData(),
    current_house_name: houseName?.value?.trim() || ""
  };

  zip.file("project.json", JSON.stringify(payload, null, 2));

  const originalPreviewJson = previewJson?.textContent || "{}";
  const originalHouseName = houseName?.value || "";
  const originalCurrentHouse = getHouseFormData();

  generateHousePreview();
  await nextFrame();
  zip.file("house-preview.png", getRendererPreviewBase64(), { base64: true });

  for (let i = 0; i < savedHousesData.length; i++) {
    const houseItem = savedHousesData[i];
    if (houseName) houseName.value = houseItem.name || "";
    applyHouseParams(houseItem.data);
    generateHousePreview();
    await nextFrame();

    const fileName = `houses/${sanitizeFilename(houseItem.name || houseItem.id || `house_${i + 1}`)}.png`;
    zip.file(fileName, getRendererPreviewBase64(), { base64: true });
  }

  for (let i = 0; i < savedModulesData.length; i++) {
    const mod = savedModulesData[i];
    renderModulePreview(mod);
    await nextFrame();

    const fileName = `modules/${sanitizeFilename(mod.name || mod.id || `module_${i + 1}`)}.png`;
    zip.file(fileName, getRendererPreviewBase64(), { base64: true });
  }

  if (houseName) houseName.value = payload.current_house_name || originalHouseName;
  applyHouseParams(originalCurrentHouse);
  generateHousePreview();
  if (previewJson) previewJson.textContent = originalPreviewJson;

  const blob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "panel-project.zip";
  a.click();
  URL.revokeObjectURL(url);
}

async function importProjectZip(file) {
  const zip = await JSZip.loadAsync(file);
  const projectFile = zip.file("project.json");

  if (!projectFile) throw new Error("project.json not found in ZIP");

  const text = await projectFile.async("text");
  const data = JSON.parse(text);

  if (data.type !== "panel_project") throw new Error("Invalid project file");

  savedModulesData = Array.isArray(data.modules) ? data.modules : [];
  savedHousesData = Array.isArray(data.houses) ? data.houses : [];

  renderSavedModules();
  populateModuleSelectors();
  renderSavedHouses();

  if (houseName) houseName.value = data.current_house_name || "";

  if (data.current_house) {
    applyHouseParams(data.current_house);
    generateHousePreview();
  }

  setActiveTab("builder");
}

// ================= SERVER ANALYZE (JSON) =================
async function analyzeModuleTextOnServer(text) {
  const response = await fetch(`${SERVER_URL}/api/parse-module`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: text,
      module_type: $("moduleType").value
    })
  });

  if (!response.ok) throw new Error('Parse failed');
  return await response.json();
}

async function analyzeHouseTextOnServer(text) {
  return fetchJson(`${SERVER_URL}/api/analyze-building-text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
}

// ================= EVENTS =================
tabModules?.addEventListener("click", () => setActiveTab("modules"));
tabLibrary?.addEventListener("click", () => setActiveTab("library"));
tabBuilder?.addEventListener("click", () => setActiveTab("builder"));

libraryFilter?.addEventListener("change", renderSavedModules);
librarySort?.addEventListener("change", renderSavedModules);
houseSort?.addEventListener("change", renderSavedHouses);

moduleAdvancedToggle?.addEventListener("click", () => {
  toggleAdvanced(moduleAdvancedToggle, moduleAdvancedContent);
});

houseAdvancedToggle?.addEventListener("click", () => {
  toggleAdvanced(houseAdvancedToggle, houseAdvancedContent);
});

moduleType?.addEventListener("change", () => {
  applyModuleTypeDefaults(moduleType.value);
  liveUpdateModulePreview();
});

moduleDescription?.addEventListener("input", () => {
  const parsed = parseModuleTextLocally(moduleDescription.value);
  applyLocalModuleParse(parsed);
});

houseDescription?.addEventListener("input", () => {
  const parsed = parseHouseTextLocally(houseDescription.value);
  applyLocalHouseParse(parsed);
});

analyzeModuleBtn?.addEventListener("click", async () => {
  const text = moduleDescription.value.trim();
  if (!text) {
    showToast("Enter module description first.", "error");
    return;
  }

  await withLoading(analyzeModuleBtn, "Analyzing...", async () => {
    try {
      // Call new API
      const response = await fetch(`${SERVER_URL}/api/parse-module`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          module_type: $("moduleType").value
        })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error);

        // AUTO-SELECT MODULE TYPE from API
        $("moduleType").value = data.module_type;
        applyModuleTypeDefaults(data.module_type);

        // Apply parameters
        if (data.params.width) $("moduleWidth").value = data.params.width;
        if (data.params.height) $("moduleHeight").value = data.params.height;
        if (data.params.depth) $("moduleDepth").value = data.params.depth;
        if (data.params.color) $("moduleColor").value = data.params.color;

      renderModulePreview(getModuleFormData());
      showToast("Module analysis completed.", "success");
    } catch (err) {
      showToast(`Analyze failed: ${err.message}`, "error");
    }
  });
});

generateModuleBtn?.addEventListener("click", async () => {
  const text = moduleDescription.value.trim();
  const moduleType = $("moduleType").value;

  // Если нет текста - генерируем из слайдеров
  let finalText = text || `${moduleType} width ${$("moduleWidth").value}m height ${$("moduleHeight").value}m depth ${$("moduleDepth").value}m`;

  generateModuleBtn.disabled = true;
  generateModuleBtn.textContent = "⏳ Generating...";

  try {
    const response = await fetch(`${SERVER_URL}/api/generate-module`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: finalText,
        module_type: moduleType
      })
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.error);

    showToast(`Module generated: ${data.module_name}`, "success");

    // === ЗАГРУЖАЕМ РЕАЛЬНЫЙ OBJ ФАЙЛ В THREE.JS ===
    if (data.obj_url) {
      await loadObjInPreview(data.obj_url);
    } else if (data.module_id) {
      // Если нет прямого URL, строим его
      const objUrl = `/modules/${moduleType}/${data.module_id}/${moduleType}.obj`;
      await loadObjInPreview(objUrl);
    }

    // Только очищаем description, name оставляем
    moduleDescription.value = '';

  } catch (err) {
    showToast(`Generation failed: ${err.message}`, "error");
  } finally {
    generateModuleBtn.disabled = false;
    generateModuleBtn.textContent = "Generate Module";
  }
});

// === ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ OBJ В THREE.JS ===
async function loadObjInPreview(objUrl) {
  return new Promise((resolve, reject) => {
    const objectsToRemove = [];
    scene.children.forEach((child) => {
      if (child !== groundPlane && !(child instanceof THREE.GridHelper)) {
        objectsToRemove.push(child);
      }
    });
    objectsToRemove.forEach(obj => scene.remove(obj));

    const loader = new window.THREE.OBJLoader();

    loader.load(
      objUrl,
      (obj) => {
        console.log(`✓ OBJ загружен: ${objUrl}`);

        if (!scene) {
          console.warn("Scene не инициализирована");
          resolve();
          return;
        }

        // Удаляем старые объекты (кроме grid и groundPlane)
        const objectsToRemove = [];
        scene.children.forEach((child) => {
          if (child !== groundPlane && !(child instanceof THREE.GridHelper)) {
            objectsToRemove.push(child);
          }
        });
        objectsToRemove.forEach(obj => scene.remove(obj));

        // Добавляем новый OBJ
        scene.add(obj);

        // Центрируем камеру на объект
        const box = new THREE.Box3().setFromObject(obj);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());

        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = camera.fov * (Math.PI / 180);
        let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));

        camera.position.copy(center);
        camera.position.z += cameraZ * 1.5;
        camera.lookAt(center);

        controls.target.copy(center);
        controls.update();

        console.log(`✓ Объект добавлен в сцену`);
        resolve();
      },
      undefined,
      (error) => {
        console.error(`Ошибка загрузки OBJ: ${error}`);
        showToast(`Failed to load 3D model: ${error.message}`, "error");
        reject(error);
      }
    );
  });
}

saveModuleBtn?.addEventListener("click", async () => {
  const text = moduleDescription.value.trim();
  const moduleType = $("moduleType").value;

  if (!text) {
    showToast("Enter module description first.", "error");
    return;
  }

  await withLoading(saveModuleBtn, "Saving...", async () => {
    try {
      const result = await fetch(`${SERVER_URL}/api/generate-module`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          module_type: moduleType
        })
      });

      const data = await result.json();
      if (!result.ok) throw new Error(data.error);

      showToast(`Module saved: ${data.module_name}`, "success");
      moduleDescription.value = '';

      // Обновляем список модулей в библиотеке и селекторах
      await loadSavedModules();
      renderSavedModules();
      populateModuleSelectors();
      setActiveTab("library");
    } catch (err) {
      showToast(`Save failed: ${err.message}`, "error");
    }
  });
});

analyzeHouseBtn?.addEventListener("click", async () => {
  const text = houseDescription.value.trim();
  if (!text) {
    showToast("Enter house description first.", "error");
    return;
  }

  const localParsed = parseHouseTextLocally(text);
  applyLocalHouseParse(localParsed);

  await withLoading(analyzeHouseBtn, "Analyzing...", async () => {
    try {
      const parsed = await analyzeHouseTextOnServer(text);
      applyHouseParams(parsed);
      generateHousePreview();
      showToast("House analysis completed.", "success");
    } catch (err) {
      showToast(`Server analyze failed. Local parse was used. ${err.message}`, "info");
    }
  });
});

generateHouseBtn?.addEventListener("click", generateHousePreview);

animateHouseBtn?.addEventListener("click", async () => {
  await withLoading(animateHouseBtn, "Animating...", async () => {
    await animateHousePreview();
  });
});

randomizeHouseBtn?.addEventListener("click", randomizeHouse);
randomizeFacadeBtn?.addEventListener("click", randomizeFacade);

saveHouseBtn?.addEventListener("click", () => {
  const validation = validateHouseForm(true);
  if (!validation.valid) {
    showToast("Please fix house form errors before saving.", "error");
    return;
  }

  const project = getCurrentHouseProject();
  savedHousesData.push(project);
  renderSavedHouses();
  showToast("House saved successfully.", "success");
});

exportLibraryZipBtn?.addEventListener("click", async () => {
  await withLoading(exportLibraryZipBtn, "Exporting...", async () => {
    try {
      await exportLibraryZip();
      showToast("Library exported successfully.", "success");
    } catch (err) {
      showToast(`Library export failed: ${err.message}`, "error");
    }
  });
});

importLibraryZipBtn?.addEventListener("click", () => {
  importLibraryZipInput?.click();
});

importLibraryZipInput?.addEventListener("change", async e => {
  const file = e.target.files?.[0];
  if (!file) return;

  await withLoading(importLibraryZipBtn, "Importing...", async () => {
    try {
      await importLibraryZip(file);
      showToast("Library imported successfully.", "success");
    } catch (err) {
      showToast(`Library import failed: ${err.message}`, "error");
    }
  });

  e.target.value = "";
});

exportProjectZipBtn?.addEventListener("click", async () => {
  await withLoading(exportProjectZipBtn, "Exporting...", async () => {
    try {
      await exportProjectZip();
      showToast("Project exported successfully.", "success");
    } catch (err) {
      showToast(`Project export failed: ${err.message}`, "error");
    }
  });
});

importProjectZipBtn?.addEventListener("click", () => {
  importProjectZipInput?.click();
});

importProjectZipInput?.addEventListener("change", async e => {
  const file = e.target.files?.[0];
  if (!file) return;

  await withLoading(importProjectZipBtn, "Importing...", async () => {
    try {
      await importProjectZip(file);
      showToast("Project imported successfully.", "success");
    } catch (err) {
      showToast(`Project import failed: ${err.message}`, "error");
    }
  });

  e.target.value = "";
});

resetBtn?.addEventListener("click", () => {
  moduleDescription.value = "";
  moduleName.value = "";
  moduleType.value = "wall";
  moduleColor.value = DEFAULTS.colors.wall;
  applyModuleTypeDefaults("wall");

  houseDescription.value = "";
  floorsInput.value = 8;
  sectionsInput.value = 3;
  widthInput.value = 18;
  depthInput.value = 2;
  if (facadeTextureSelect) facadeTextureSelect.value = "";
  if (facadeTextureRepeat) facadeTextureRepeat.value = DEFAULTS.facade.textureScale;
  hasBalconies.checked = true;
  balconyRate.value = 0.25;
  windowCols.value = 8;
  updateHouseRangeLabels();

  clearModuleValidation();
  clearHouseValidation();

  if (previewJson) previewJson.textContent = "{}";
  if (houseName) houseName.value = "";

  clearSceneMeshes();
  camera.position.set(18, 12, 18);
  controls.target.set(0, 4, 0);
  controls.update();

  if (!modulePage?.classList.contains("hidden")) {
    liveUpdateModulePreview();
  }

  showToast("Form reset completed.", "info");
});

[
  moduleName,
  moduleColor,
  moduleWidth,
  moduleHeight,
  moduleDepth
].forEach(input => {
  input?.addEventListener("input", () => {
    validateModuleForm(false);
    liveUpdateModulePreview();
  });
  input?.addEventListener("change", () => {
    validateModuleForm(false);
    liveUpdateModulePreview();
  });
});

[
  floorsInput,
  sectionsInput,
  widthInput,
  depthInput,
  balconyRate,
  windowCols,
  facadeTextureRepeat
].forEach(input => {
  input?.addEventListener("input", () => {
    updateHouseRangeLabels();
    updateHouseJsonPreview();
    validateHouseForm(false);
  });
});

[
  facadeTextureSelect,
  hasBalconies,
  wallModule,
  windowModule,
  doorModule,
  roofModule,
  balconyModule
].forEach(input => {
  input?.addEventListener("input", () => {
    updateHouseJsonPreview();
    validateHouseForm(false);
  });

  input?.addEventListener("change", () => {
    updateHouseJsonPreview();
    validateHouseForm(false);
    if (!builderPage?.classList.contains("hidden")) {
      const validation = validateHouseForm(false);
      if (validation.valid) generateHousePreview();
    }
  });
});

facadeTextureRepeat?.addEventListener("input", () => {
  updateHouseRangeLabels();
  updateHouseJsonPreview();
  validateHouseForm(false);
  if (!builderPage?.classList.contains("hidden")) {
    const validation = validateHouseForm(false);
    if (validation.valid) generateHousePreview();
  }
});

// ================= POINTER =================
function handlePointerLeave() {
  clearHoverState();
  hoverInfo?.classList.add("hidden");
}

function onPointerMove(event) {
  if (!interactiveObjects.length) return;

  const rect = preview.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(interactiveObjects, false);
  const hit = intersects[0]?.object || null;

  if (lastHovered?.object !== hit) {
    clearHoverState();
  }

  if (hit) {
    if (!lastHovered || lastHovered.object !== hit) {
      lastHovered = {
        object: hit,
        baseColor: hit.userData.baseColor?.clone?.() || new window.THREE.Color(0xffffff)
      };
      hit.material.color.set(0xb87cff);
    }

    hoverInfo?.classList.remove("hidden");
    if (hoverInfo) hoverInfo.textContent = hit.userData.label || "Element";
  } else {
    hoverInfo?.classList.add("hidden");
  }
}

// ================= INIT =================
(async () => {
  await initScene();
  updateHouseRangeLabels();
  await loadFacadeTextures();
  await loadSavedModules();
  renderSavedHouses();
  setActiveTab("modules");

  floorsInput.value = 8;
  sectionsInput.value = 3;
  widthInput.value = 18;
  depthInput.value = 2;
  if (facadeTextureSelect) facadeTextureSelect.value = "";
  hasBalconies.checked = true;
  balconyRate.value = 0.25;
  windowCols.value = 8;
  if (facadeTextureRepeat) facadeTextureRepeat.value = DEFAULTS.facade.textureScale;

  updateHouseRangeLabels();
  applyModuleTypeDefaults(moduleType.value);
  liveUpdateModulePreview();
})();