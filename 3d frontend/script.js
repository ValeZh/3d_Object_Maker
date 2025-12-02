<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>3D Object Maker</title>
  <link rel="stylesheet" href="style.css"/>
</head>
<body>

    <div class="app">
    <header class="top-bar">
      <div class="logo">
        <img src="photo_2025-11-18_14-06-36.jpg" alt="logo" />
        <span>3D Object Maker</span>
      </div>
      <div class="controls">
        <button id="reset" class="control-btn">Reset</button>
      </div>
    </header>

    <main class="content">
      <aside class="params">
        <h2>3D Object Parameters</h2>

        <label class="field-label">Shape:</label>
        <div class="dropdown" id="shapeDropdown">
          <button class="dropbtn">Select Shape</button>
          <div class="dropdown-content" id="shapeOptions"></div>
        </div>

        <label class="field-label">Color:</label>
        <input id="color" type="color" />

        <label class="field-label">Texture:</label>
        <div class="dropdown" id="textureDropdown">
          <button class="dropbtn">Select Texture</button>
          <div class="dropdown-content" id="textureOptions"></div>
        </div>

        <label class="field-label">AI Description:</label>
        <textarea id="description" rows="4" placeholder="Example: red metallic cube"></textarea>
        <button id="generate" class="export">Generate</button>
        <button id="export" class="export">Export OBJ</button>
      </aside>

      <section class="preview">
        <div id="objectPreview"></div>
      </section>
    </main>
  </div>

  <!-- Three.js ES Modules + экспорт в window -->
  <script type="importmap">
  {
    "imports": {
      "three": "https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/"
    }
  }
  </script>

  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
    import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';
    import { OBJExporter } from 'three/addons/exporters/OBJExporter.js';

    // Делаем THREE расширяемым и экспортируем loaders
    window.THREE = Object.assign({}, THREE);
    window.THREE.OrbitControls = OrbitControls;
    window.THREE.OBJLoader = OBJLoader;
    window.THREE.MTLLoader = MTLLoader;
    window.THREE.OBJExporter = OBJExporter;

    console.log("Three.js + loaders загружены");
  </script>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
  <script src="script.js"></script>
</body>
</html>