// Подключаем Three.js
let scene, camera, renderer, object;
let currentShape = 'Куб';
let currentColor = '#6952BE';
let currentTexture = 'plastic';

const colorSelect = document.getElementById('color');
const textureSelect = document.getElementById('texture');
const shapeSelect = document.getElementById('shape');
const resetButton = document.getElementById('reset');
const saveButton = document.getElementById('save');
const container = document.getElementById('3d-container');

function init3DScene() {
  // Сцена
  scene = new THREE.Scene();
  // Камера
  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  // Рендерер
  renderer = new THREE.WebGLRenderer();
  renderer.setSize(window.innerWidth, window.innerHeight);
  container.appendChild(renderer.domElement);

  // Инициализация объекта
  createObject(currentShape, currentColor, currentTexture);

  // Камера
  camera.position.z = 5;

  // Функция анимации
  function animate() {
    requestAnimationFrame(animate);
    if (object) {
      object.rotation.x += 0.01;
      object.rotation.y += 0.01;
    }
    renderer.render(scene, camera);
  }

  animate();
}

function createObject(shape, color, texture) {
  if (object) {
    scene.remove(object); // Удаляем старый объект
  }

  const material = new THREE.MeshBasicMaterial({ color: color });

  if (texture === 'metal') {
    material.map = new THREE.TextureLoader().load('metal-texture.jpg');
  } else if (texture === 'wood') {
    material.map = new THREE.TextureLoader().load('wood-texture.jpg');
  }

  let geometry;

  switch (shape) {
    case 'Куб':
      geometry = new THREE.BoxGeometry(1, 1, 1);
      break;
    case 'Сфера':
      geometry = new THREE.SphereGeometry(1, 32, 32);
      break;
    case 'Пирамида':
      geometry = new THREE.ConeGeometry(1, 1, 4);
      break;
    case 'Призма':
      geometry = new THREE.CylinderGeometry(1, 1, 1, 4);
      break;
    default:
      geometry = new THREE.BoxGeometry(1, 1, 1);
      break;
  }

  object = new THREE.Mesh(geometry, material);
  scene.add(object);
}

// Обновляем объект при изменении параметров
function updateObject() {
  currentShape = shapeSelect.value;
  currentColor = colorSelect.value;
  currentTexture = textureSelect.value;
  createObject(currentShape, currentColor, currentTexture);
}

colorSelect.addEventListener('change', updateObject);
textureSelect.addEventListener('change', updateObject);
shapeSelect.addEventListener('change', updateObject);

// Кнопка сброса
resetButton.addEventListener('click', () => {
  colorSelect.value = '#6952BE';
  textureSelect.value = 'plastic'; })