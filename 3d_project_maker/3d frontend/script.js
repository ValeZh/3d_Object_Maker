const colorSelect = document.getElementById('color');
const textureSelect = document.getElementById('texture');
const resetButton = document.getElementById('reset');
const shapeSelect = document.getElementById('shape');
const objectDiv = document.getElementById('object');

function updateObject() {
  const shape = shapeSelect.value;
  const color = colorSelect.value;
  const texture = textureSelect.value;


  objectDiv.className = '';
  objectDiv.style.background = '';
  objectDiv.style.borderBottomColor = '';
  objectDiv.innerHTML = '';

  if(shape === 'Куб') {
    objectDiv.classList.add('cube');
    objectDiv.innerHTML = `
      <div class="face front"></div>
      <div class="face back"></div>
      <div class="face left"></div>
      <div class="face right"></div>
      <div class="face top"></div>
      <div class="face bottom"></div>
    `;
    objectDiv.querySelectorAll('.face').forEach(face => {
      face.style.background = `linear-gradient(135deg, ${color}, #5AD1E7)`;
      if(texture === 'metal') face.style.filter = 'brightness(1.3)';
      else if(texture === 'wood') face.style.filter = 'brightness(0.8)';
      else face.style.filter = 'brightness(1)';
    });
  }
  else if(shape === 'Сфера') {
    objectDiv.classList.add('sphere');
    objectDiv.style.background = `radial-gradient(circle at 30% 30%, ${color}, #5AD1E7)`;
  }
  else if(shape === 'Пирамида') {
    objectDiv.classList.add('pyramid');
    objectDiv.style.borderBottomColor = color;
  }
  else if(shape === 'Призма') {
    objectDiv.classList.add('prism');
    objectDiv.style.background = `linear-gradient(135deg, ${color}, #5AD1E7)`;
  }
}


colorSelect.addEventListener('change', updateObject);
textureSelect.addEventListener('change', updateObject);
shapeSelect.addEventListener('change', updateObject);
resetButton.addEventListener('click', () => {
  colorSelect.value = '#6952BE';
  textureSelect.value = 'plastic';
  shapeSelect.value = 'Куб';
  updateObject();
});


