// https://github.com/MrForExample/ComfyUI-3D-Pack/blob/main/web/js/threeVisualizer.js
import * as THREE from "three";
import { api } from "../../../scripts/api.js";

import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

export function getRGBValue(colorString = "rgb(0, 0, 0)", scale01 = false) {
  var color = colorString.split(/[\D]+/).filter(Boolean);
  for (let index = 0; index < color.length; index++) {
    color[index] = Number(color[index]);
  }

  if (scale01) {
    for (let index = 0; index < color.length; index++) {
      color[index] = color[index] / 255;
    }
  }

  return color;
}

const visualizer = document.getElementById("visualizer");
const container = document.getElementById("container");
const progressDialog = document.getElementById("progress-dialog");
const progressIndicator = document.getElementById("progress-indicator");
// const colorPicker = document.getElementById("color-picker");

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const pmremGenerator = new THREE.PMREMGenerator(renderer);

// scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
scene.environment = pmremGenerator.fromScene(
  new RoomEnvironment(renderer),
  0.04
).texture;

const ambientLight = new THREE.AmbientLight(0xffffff);

const camera = new THREE.PerspectiveCamera(
  40,
  window.innerWidth / window.innerHeight,
  1,
  100
);
camera.position.set(1.5, 0, 0);
camera.rotation.set(-1.4, 1.4, 1.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.update();
controls.enablePan = true;
controls.enableDamping = true;

// Handle window reseize event
window.onresize = function () {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
};

var lastFilepath = "";
var needUpdate = false;

const onProgress = function (xhr) {
  if (xhr.lengthComputable) {
    progressIndicator.value = (xhr.loaded / xhr.total) * 100;
  }
};

const onError = function (e) {
  console.error(e);
};

function animate() {
  var filepath = visualizer.getAttribute("filepath");
  if (filepath == lastFilepath) {
    if (needUpdate) {
      controls.update();
      renderer.render(scene, camera);
    }
    requestAnimationFrame(animate);
  } else {
    needUpdate = false;
    scene.clear();
    progressDialog.open = true;
    lastFilepath = filepath;
    main(JSON.parse(lastFilepath));
  }
}

async function main(params) {
  if (params?.filename) {
    const url = api
      .apiURL("/view?" + new URLSearchParams(params))
      .replace(/extensions.*\//, "");
    const fileExt = params.filename.slice(params.filename.lastIndexOf(".") + 1);

    if (fileExt == "glb") {
      const dracoLoader = new DRACOLoader();
      dracoLoader.setDecoderPath(
        "https://unpkg.com/three@latest/examples/jsm/libs/draco/gltf/"
      );
      const loader = new GLTFLoader();
      loader.setDRACOLoader(dracoLoader);

      loader.load(
        url,
        function (gltf) {
          const model = gltf.scene;
          model.position.set(0, 0, 0);
          model.scale.set(1, 1, 1);
          scene.add(model);
        },
        onProgress,
        onError
      );
    }

    needUpdate = true;
  } else {
    scene.clear();
  }

  progressDialog.close();
  animate();
}

main();
