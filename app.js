import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19.2/dist/lil-gui.esm.min.js';


const MAX_STEPS = 512;


/* -------------------------------------------------------------------------- */
/* GLSL SHADERS                                                               */
/* -------------------------------------------------------------------------- */

const VERTEX_SHADER = /* glsl */ `
precision highp float;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

in vec3 position;

out vec3 vLocalPosition;

void main() {
    vLocalPosition = position;

    gl_Position = projectionMatrix
                * modelViewMatrix
                * vec4(position, 1.0);
}
`;


const FRAGMENT_SHADER = /* glsl */ `
precision highp float;
precision highp sampler3D;

uniform sampler3D uVolume;
uniform sampler2D uColorMap;

uniform vec3 uCameraLocal;
uniform vec3 uExtent;

uniform vec3 uClipMin;
uniform vec3 uClipMax;

uniform float uReferenceLength;

uniform float uLower;
uniform float uUpper;
uniform float uSoftness;
uniform float uGlobalOpacity;
uniform float uOpticalDensity;
uniform float uGamma;

uniform int uSteps;

in vec3 vLocalPosition;

out vec4 outColor;


/*
  Intersect a ray with an axis-aligned box.

  Returns:
      x = entry distance
      y = exit distance
*/
vec2 intersectBox(
    vec3 rayOrigin,
    vec3 rayDirection,
    vec3 boxMin,
    vec3 boxMax
) {
    vec3 safeDirection = rayDirection;

    if (abs(safeDirection.x) < 1.0e-6) safeDirection.x = 1.0e-6;
    if (abs(safeDirection.y) < 1.0e-6) safeDirection.y = 1.0e-6;
    if (abs(safeDirection.z) < 1.0e-6) safeDirection.z = 1.0e-6;

    vec3 invDirection = 1.0 / safeDirection;

    vec3 t0 = (boxMin - rayOrigin) * invDirection;
    vec3 t1 = (boxMax - rayOrigin) * invDirection;

    vec3 tMin = min(t0, t1);
    vec3 tMax = max(t0, t1);

    float entry = max(max(tMin.x, tMin.y), tMin.z);
    float exit  = min(min(tMax.x, tMax.y), tMax.z);

    return vec2(entry, exit);
}


void main() {
    /*
      The box is rendered using back faces. For each pixel,
      vLocalPosition points toward the rear surface of the volume.
    */
    vec3 rayDirection = normalize(vLocalPosition - uCameraLocal);

    /*
      Crop range is stored in normalized texture coordinates [0, 1].
      Convert it to local physical coordinates.
    */
    vec3 boxMin = (uClipMin - vec3(0.5)) * uExtent;
    vec3 boxMax = (uClipMax - vec3(0.5)) * uExtent;

    vec2 hit = intersectBox(
        uCameraLocal,
        rayDirection,
        boxMin,
        boxMax
    );

    float tStart = max(hit.x, 0.0);
    float tEnd   = hit.y;

    if (tEnd <= tStart) {
        discard;
    }

    float stepLength = (tEnd - tStart) / float(uSteps);

    /*
      accum.rgb is premultiplied color during ray marching.
      accum.a is accumulated opacity.
    */
    vec4 accum = vec4(0.0);

    for (int i = 0; i < ${MAX_STEPS}; ++i) {
        if (i >= uSteps) {
            break;
        }

        float t = tStart + (float(i) + 0.5) * stepLength;

        vec3 samplePosition = uCameraLocal + rayDirection * t;

        /*
          Convert physical local position to texture coordinates [0, 1].
        */
        vec3 texCoord = samplePosition / uExtent + vec3(0.5);

        texCoord = clamp(texCoord, vec3(0.0), vec3(1.0));

        /*
          uVolume is an 8-bit R texture, but WebGL returns a normalized
          floating-point sample in [0, 1].
        */
        float density = texture(uVolume, texCoord).r;

        /*
          Density transfer function.

          mapped = 0 at lower cutoff
          mapped = 1 at upper/saturation value
        */
        float mapped = clamp(
            (density - uLower) / max(uUpper - uLower, 1.0e-5),
            0.0,
            1.0
        );

        /*
          Soft transition at the threshold rather than a hard cutoff.
        */
        float visible = smoothstep(
            0.0,
            max(uSoftness, 1.0e-5),
            mapped
        );

        /*
          Gamma changes mid-density visibility:

          gamma > 1: emphasizes dense structures
          gamma < 1: reveals diffuse structures
        */
        float extinction = visible
            * pow(max(mapped, 1.0e-5), max(uGamma, 0.01));

        /*
          Beer-Lambert style optical absorption.

          Including stepLength makes opacity approximately stable when
          changing the number of ray-marching steps.
        */
        float sampleAlpha = 1.0 - exp(
            -uOpticalDensity
            * extinction
            * (stepLength / uReferenceLength)
        );

        if (sampleAlpha > 1.0e-6) {
            vec3 sampleColor = texture(
                uColorMap,
                vec2(mapped, 0.5)
            ).rgb;

            float remaining = 1.0 - accum.a;

            accum.rgb += remaining * sampleAlpha * sampleColor;
            accum.a   += remaining * sampleAlpha;
        }

        /*
          Early ray termination: no need to sample after opacity is high.
        */
        if (accum.a > 0.995) {
            break;
        }
    }

    float finalAlpha = accum.a * uGlobalOpacity;

    if (finalAlpha < 0.002) {
        discard;
    }

    /*
      accum.rgb is premultiplied. Convert back to straight RGB because
      Three.js uses ordinary alpha blending for transparent materials.
    */
    vec3 straightColor = accum.rgb / max(accum.a, 1.0e-6);

    outColor = vec4(straightColor, finalAlpha);
}
`;


/* -------------------------------------------------------------------------- */
/* INITIALIZATION                                                             */
/* -------------------------------------------------------------------------- */

const app = document.getElementById('app');
const status = document.getElementById('status');
const errorBox = document.getElementById('error');

const renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: 'high-performance',
});

renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x02050a, 1);

app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x02050a);

let camera;
let controls;

let volumeMesh;
let uniforms;

let outerOutline;
let clipOutline;

/*
  The 3-D Gould Belt ellipse and its material.
  Keeping the material as a separate variable allows the GUI
  to update opacity and line width immediately.
*/
let gouldBeltLine;
let gouldBeltMaterial;

let extent;
let centre;
let ranges;
let initialCameraPosition;

let params;

let renderPending = false;

window.addEventListener('resize', onResize);

if (!renderer.capabilities.isWebGL2) {
    reportError(
        new Error(
            'This viewer requires WebGL 2. Try a current version of Chrome, Firefox, Edge, or Safari.'
        )
    );
} else {
    initialise().catch(reportError);
}


/* -------------------------------------------------------------------------- */
/* LOAD DATA                                                                  */
/* -------------------------------------------------------------------------- */

async function initialise() {
    const [metadata, rawBuffer, colorMap] = await Promise.all([
        loadJSON('./data/density.json'),
        loadArrayBuffer('./data/density.u8'),
        new THREE.TextureLoader().loadAsync('./data/freeze.png'),
    ]);

    if (
        !Array.isArray(metadata.dimensions)
        || metadata.dimensions.length !== 3
    ) {
        throw new Error('density.json has no valid "dimensions" field.');
    }

    const [nx, ny, nz] = metadata.dimensions.map(Number);

    if (
        ![nx, ny, nz].every(
            value => Number.isInteger(value) && value > 1
        )
    ) {
        throw new Error('Invalid texture dimensions in density.json.');
    }

    const expectedBytes = nx * ny * nz;

    if (rawBuffer.byteLength !== expectedBytes) {
        throw new Error(
            `density.u8 has ${rawBuffer.byteLength} bytes, but `
            + `${expectedBytes} bytes were expected.`
        );
    }

    const gl = renderer.getContext();
    const maxTextureSize3D = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);

    if (Math.max(nx, ny, nz) > maxTextureSize3D) {
        throw new Error(
            `Your 3-D texture is too large for this GPU. `
            + `Maximum supported dimension: ${maxTextureSize3D}`
        );
    }

    ranges = {
        x: readRange(metadata.bounds?.x, 'bounds.x'),
        y: readRange(metadata.bounds?.y, 'bounds.y'),
        z: readRange(metadata.bounds?.z, 'bounds.z'),
    };

    extent = new THREE.Vector3(
        ranges.x[1] - ranges.x[0],
        ranges.y[1] - ranges.y[0],
        ranges.z[1] - ranges.z[0],
    );

    centre = new THREE.Vector3(
        0.5 * (ranges.x[0] + ranges.x[1]),
        0.5 * (ranges.y[0] + ranges.y[1]),
        0.5 * (ranges.z[0] + ranges.z[1]),
    );

    const maxExtent = Math.max(extent.x, extent.y, extent.z);

    /*
      Create the 3-D scalar texture.
    */
    const volumeData = new Uint8Array(rawBuffer);

    const volumeTexture = new THREE.Data3DTexture(
        volumeData,
        nx,
        ny,
        nz
    );

    volumeTexture.format = THREE.RedFormat;
    volumeTexture.type = THREE.UnsignedByteType;

    volumeTexture.minFilter = THREE.LinearFilter;
    volumeTexture.magFilter = THREE.LinearFilter;

    volumeTexture.wrapS = THREE.ClampToEdgeWrapping;
    volumeTexture.wrapT = THREE.ClampToEdgeWrapping;
    volumeTexture.wrapR = THREE.ClampToEdgeWrapping;

    volumeTexture.unpackAlignment = 1;
    volumeTexture.generateMipmaps = false;
    volumeTexture.flipY = false;
    volumeTexture.colorSpace = THREE.NoColorSpace;

    volumeTexture.needsUpdate = true;

    /*
      The LUT is already a display colormap, so keep its RGB values raw.
    */
    colorMap.colorSpace = THREE.NoColorSpace;
    colorMap.minFilter = THREE.LinearFilter;
    colorMap.magFilter = THREE.LinearFilter;
    colorMap.wrapS = THREE.ClampToEdgeWrapping;
    colorMap.wrapT = THREE.ClampToEdgeWrapping;
    colorMap.generateMipmaps = false;
    colorMap.needsUpdate = true;

    const defaults = metadata.defaultControls ?? {};

    params = {
    /*
      ----------------------------------------------------------------
      OB-star density field controls
      ----------------------------------------------------------------
    */
    showDensity: true,

    lower: finiteNumber(defaults.lower, 0.425),
    upper: finiteNumber(defaults.upper, 0.820),
    softness: finiteNumber(defaults.softness, 0.25),

    opacity: finiteNumber(defaults.opacity, 1.00),
    opticalDensity: finiteNumber(defaults.opticalDensity, 50.0),
    gamma: finiteNumber(defaults.gamma, 2.0),

    steps: Math.round(finiteNumber(defaults.steps, 170)),

    xMin: ranges.x[0],
    xMax: ranges.x[1],

    yMin: ranges.y[0],
    yMax: ranges.y[1],

    zMin: ranges.z[0],
    zMax: ranges.z[1],

    showBounds: true,
    showCropBox: false,

    /*
      ----------------------------------------------------------------
      Gould Belt model controls
      ----------------------------------------------------------------
    */
    showGouldBelt: true,

    /*
      This width is measured in screen pixels, not pc.

      Line2 supports thick lines consistently across browsers,
      unlike ordinary THREE.Line / LineBasicMaterial.
    */
    gouldBeltLineWidth: 3.0,

    /*
      0 = fully transparent
      1 = fully opaque
    */
    gouldBeltOpacity: 0.95,

    resetCrop: () => {},
    resetView: () => {},
};

    /*
      Camera.
    */
    camera = new THREE.PerspectiveCamera(
        42,
        window.innerWidth / window.innerHeight,
        Math.max(0.1, maxExtent * 0.001),
        maxExtent * 25.0
    );

    /*
      Use astronomical/Galactic orientation:

          XY = reference / Galactic plane
          +Z = vertical direction

      This is what makes orbiting naturally rotate around the XY plane.
    */
    camera.up.set(0, 0, 1);

    /*
      Initial camera position.

      The camera is mainly above the XY plane, i.e. at positive Z,
      but slightly offset in X and Y so that:

      - the camera is not exactly aligned with the +Z axis;
      - the volume is visibly 3-D on startup;
      - OrbitControls does not encounter an "up vector parallel to view"
        degeneracy.

      You can tune these three coefficients later.
    */
    initialCameraPosition = new THREE.Vector3(
        centre.x + 0.00 * maxExtent,
        centre.y - 0.15 * maxExtent,
        centre.z + 1.80 * maxExtent,
    );

    camera.position.copy(initialCameraPosition);

    controls = new OrbitControls(camera, renderer.domElement);

    controls.target.copy(centre);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;

    controls.minDistance = maxExtent * 0.02;
    controls.maxDistance = maxExtent * 15.0;

    controls.update();

    /*
      Shader uniforms.
    */
    uniforms = {
        uVolume: { value: volumeTexture },
        uColorMap: { value: colorMap },

        uCameraLocal: { value: new THREE.Vector3() },

        uExtent: { value: extent.clone() },
        uReferenceLength: { value: maxExtent },

        uClipMin: { value: new THREE.Vector3(0, 0, 0) },
        uClipMax: { value: new THREE.Vector3(1, 1, 1) },

        uLower: { value: params.lower },
        uUpper: { value: params.upper },
        uSoftness: { value: params.softness },

        uGlobalOpacity: { value: params.opacity },
        uOpticalDensity: { value: params.opticalDensity },
        uGamma: { value: params.gamma },

        uSteps: { value: params.steps },
    };

    /*
      RawShaderMaterial is used because sampler3D requires GLSL 3.
    */
    const volumeMaterial = new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,

        uniforms,

        vertexShader: VERTEX_SHADER,
        fragmentShader: FRAGMENT_SHADER,

        side: THREE.BackSide,

        transparent: true,
        depthWrite: false,
        depthTest: true,

        blending: THREE.NormalBlending,
        premultipliedAlpha: false,
    });

    /*
      The geometry physically occupies your x/y/z range in pc.
    */
    volumeMesh = new THREE.Mesh(
        new THREE.BoxGeometry(extent.x, extent.y, extent.z),
        volumeMaterial
    );

    volumeMesh.position.copy(centre);
    volumeMesh.renderOrder = 0;

    scene.add(volumeMesh);

    /*
      Outer domain outline.
    */
    outerOutline = makeOutline(0x6481a0, 0.38);
    outerOutline.scale.copy(extent);
    outerOutline.renderOrder = 2;

    volumeMesh.add(outerOutline);

    clipOutline = makeOutline(0x8fe7ff, 0.88);
    clipOutline.renderOrder = 3;

    volumeMesh.add(clipOutline);


    /*
      Create the 3-D Gould Belt ellipse.

      It is added directly to `scene`, rather than to `volumeMesh`,
      because its coordinates are already physical Galactic Cartesian
      coordinates in pc.
    */
    gouldBeltLine = createGouldBeltModel();

    gouldBeltLine.renderOrder = 5;

    scene.add(gouldBeltLine);


    createGUI();

    controls.addEventListener('change', requestRender);

    const volumeMiB = rawBuffer.byteLength / 1024**2;

    status.textContent =
        `${nx} × ${ny} × ${nz} volume texture · `
        + `${volumeMiB.toFixed(3)} MiB · `
        + `WebGL2 ray marching`;

    requestRender();
}


/* -------------------------------------------------------------------------- */
/* GUI                                                                        */
/* -------------------------------------------------------------------------- */

function createGUI() {
    const gui = new GUI({
        title: 'Scene layers',
        width: 330,
    });

    gui.domElement.style.zIndex = '20';

    /*
      Helper: update the rendering whenever a GUI value changes.
    */
    const watched = (controller) => {
        controller.onChange(requestRender);
        return controller;
    };


    /*
      ==================================================================
      1. OB STAR DENSITY FIELD
      ==================================================================
    */
    const densityFolder = gui.addFolder('OB star density field (ALS III, Pantaleoni et al. 2025)');

    watched(
        densityFolder
            .add(params, 'showDensity')
            .name('Visible')
    );

    const transferFolder = densityFolder.addFolder(
        'Transfer function'
    );

    watched(
        transferFolder
            .add(params, 'lower', 0.0, 0.995, 0.005)
            .name('Lower cutoff')
    );

    watched(
        transferFolder
            .add(params, 'upper', 0.005, 1.0, 0.005)
            .name('Upper saturation')
    );

    watched(
        transferFolder
            .add(params, 'softness', 0.001, 0.50, 0.001)
            .name('Threshold softness')
    );

    watched(
        transferFolder
            .add(params, 'opacity', 0.0, 1.0, 0.01)
            .name('Final opacity')
    );

    watched(
        transferFolder
            .add(params, 'opticalDensity', 0.0, 100.0, 0.1)
            .name('Cloud density')
    );

    watched(
        transferFolder
            .add(params, 'gamma', 0.20, 3.0, 0.02)
            .name('Contrast gamma')
    );

    transferFolder.open();


    const qualityFolder = densityFolder.addFolder(
        'Render quality'
    );

    watched(
        qualityFolder
            .add(params, 'steps', 32, MAX_STEPS, 1)
            .name('Ray-march samples')
    );


    const cropFolder = densityFolder.addFolder(
        'Crop volume [pc]'
    );

    const cropControllers = [];

    function addCropControl(key, label, range) {
        const controller = watched(
            cropFolder
                .add(
                    params,
                    key,
                    range[0],
                    range[1],
                    (range[1] - range[0]) / 200.0
                )
                .name(label)
        );

        cropControllers.push(controller);
    }

    addCropControl('xMin', 'X min [pc]', ranges.x);
    addCropControl('xMax', 'X max [pc]', ranges.x);

    addCropControl('yMin', 'Y min [pc]', ranges.y);
    addCropControl('yMax', 'Y max [pc]', ranges.y);

    addCropControl('zMin', 'Z min [pc]', ranges.z);
    addCropControl('zMax', 'Z max [pc]', ranges.z);


    const densityViewFolder = densityFolder.addFolder(
        'Volume guides'
    );

    watched(
        densityViewFolder
            .add(params, 'showBounds')
            .name('Show domain box')
    );

    watched(
        densityViewFolder
            .add(params, 'showCropBox')
            .name('Show crop box')
    );


    params.resetCrop = () => {
        params.xMin = ranges.x[0];
        params.xMax = ranges.x[1];

        params.yMin = ranges.y[0];
        params.yMax = ranges.y[1];

        params.zMin = ranges.z[0];
        params.zMax = ranges.z[1];

        cropControllers.forEach((controller) => {
            controller.updateDisplay();
        });

        requestRender();
    };

    densityFolder
        .add(params, 'resetCrop')
        .name('Reset crop');


    /*
      ==================================================================
      2. GOULD BELT MODEL
      ==================================================================
    */
    const gouldFolder = gui.addFolder('Gould Belt model (Perrot & Grenier 2003)');

    watched(
        gouldFolder
            .add(params, 'showGouldBelt')
            .name('Visible')
    );

    watched(
        gouldFolder
            .add(params, 'gouldBeltLineWidth', 1.0, 15.0, 0.25)
            .name('Line width [px]')
    );

    watched(
        gouldFolder
            .add(params, 'gouldBeltOpacity', 0.0, 1.0, 0.01)
            .name('Opacity')
    );


    /*
      ==================================================================
      GENERAL VIEW
      ==================================================================
    */
    const viewFolder = gui.addFolder('View');

    params.resetView = () => {
        camera.position.copy(initialCameraPosition);
        controls.target.copy(centre);
        controls.update();

        requestRender();
    };

    viewFolder
        .add(params, 'resetView')
        .name('Reset camera');


    /*
      Initial menu state.

      The two principal scientific layers are expanded initially.
      You can remove either `.open()` call if you prefer a more compact
      initial menu.
    */
    densityFolder.open();
    gouldFolder.open();
}


/* -------------------------------------------------------------------------- */
/* UPDATE UNIFORMS                                                            */
/* -------------------------------------------------------------------------- */

function syncUniforms() {
    const lower = THREE.MathUtils.clamp(
        finiteNumber(params.lower, 0.5),
        0.0,
        0.995
    );

    const upper = THREE.MathUtils.clamp(
        finiteNumber(params.upper, 1.0),
        lower + 0.005,
        1.0
    );

    /*
     Toggle the entire KDE volume.
   */
   volumeMesh.visible = Boolean(params.showDensity);

   /*
     Toggle the 3-D Gould Belt ellipse.
   */
   gouldBeltLine.visible = Boolean(params.showGouldBelt);

   /*
     Update Gould Belt line appearance.
   */
   gouldBeltMaterial.linewidth = THREE.MathUtils.clamp(
       finiteNumber(params.gouldBeltLineWidth, 3.0),
       0.5,
       30.0
   );

   gouldBeltMaterial.opacity = THREE.MathUtils.clamp(
       finiteNumber(params.gouldBeltOpacity, 0.95),
       0.0,
       1.0
   );

    uniforms.uLower.value = lower;
    uniforms.uUpper.value = upper;

    uniforms.uSoftness.value = THREE.MathUtils.clamp(
        finiteNumber(params.softness, 0.035),
        0.001,
        0.5
    );

    uniforms.uGlobalOpacity.value = THREE.MathUtils.clamp(
        finiteNumber(params.opacity, 0.8),
        0.0,
        1.0
    );

    uniforms.uOpticalDensity.value = THREE.MathUtils.clamp(
        finiteNumber(params.opticalDensity, 50.0),
        0.0,
        80.0
    );

    uniforms.uGamma.value = THREE.MathUtils.clamp(
        finiteNumber(params.gamma, 0.8),
        0.05,
        5.0
    );

    uniforms.uSteps.value = Math.round(
        THREE.MathUtils.clamp(
            finiteNumber(params.steps, 128),
            8,
            MAX_STEPS
        )
    );

    const [x0, x1] = stableInterval(
        params.xMin,
        params.xMax,
        ranges.x[0],
        ranges.x[1]
    );

    const [y0, y1] = stableInterval(
        params.yMin,
        params.yMax,
        ranges.y[0],
        ranges.y[1]
    );

    const [z0, z1] = stableInterval(
        params.zMin,
        params.zMax,
        ranges.z[0],
        ranges.z[1]
    );

    const clipMinX = (x0 - ranges.x[0]) / (ranges.x[1] - ranges.x[0]);
    const clipMaxX = (x1 - ranges.x[0]) / (ranges.x[1] - ranges.x[0]);

    const clipMinY = (y0 - ranges.y[0]) / (ranges.y[1] - ranges.y[0]);
    const clipMaxY = (y1 - ranges.y[0]) / (ranges.y[1] - ranges.y[0]);

    const clipMinZ = (z0 - ranges.z[0]) / (ranges.z[1] - ranges.z[0]);
    const clipMaxZ = (z1 - ranges.z[0]) / (ranges.z[1] - ranges.z[0]);

    uniforms.uClipMin.value.set(
        clipMinX,
        clipMinY,
        clipMinZ
    );

    uniforms.uClipMax.value.set(
        clipMaxX,
        clipMaxY,
        clipMaxZ
    );

    /*
      Update visible crop box.
    */
    clipOutline.position.set(
        ((clipMinX + clipMaxX) * 0.5 - 0.5) * extent.x,
        ((clipMinY + clipMaxY) * 0.5 - 0.5) * extent.y,
        ((clipMinZ + clipMaxZ) * 0.5 - 0.5) * extent.z
    );

    clipOutline.scale.set(
        (clipMaxX - clipMinX) * extent.x,
        (clipMaxY - clipMinY) * extent.y,
        (clipMaxZ - clipMinZ) * extent.z
    );

    /*
      The domain/crop guides belong to the density-field layer.
      They disappear when "OB star density field" is switched off.
    */
    outerOutline.visible =
        Boolean(params.showDensity) &&
        Boolean(params.showBounds);

    clipOutline.visible =
        Boolean(params.showDensity) &&
        Boolean(params.showCropBox);
    }


/* -------------------------------------------------------------------------- */
/* RENDERING                                                                  */
/* -------------------------------------------------------------------------- */

function requestRender() {
    if (!camera || !volumeMesh || renderPending) {
        return;
    }

    renderPending = true;
    requestAnimationFrame(render);
}


function render() {
    renderPending = false;

    controls.update();

    syncUniforms();

    scene.updateMatrixWorld();
    camera.updateMatrixWorld();

    /*
      Shader calculations occur in volume-local coordinates.
    */
    uniforms.uCameraLocal.value.copy(camera.position);
    volumeMesh.worldToLocal(uniforms.uCameraLocal.value);

    renderer.render(scene, camera);
}


/* -------------------------------------------------------------------------- */
/* HELPERS                                                                    */
/* -------------------------------------------------------------------------- */

function createGouldBeltModel() {
    /*
      Gould Belt model parameters, in pc and degrees.

      These are the parameters you used in Python:

          a       = 373 pc
          b       = 233 pc
          d_c     = 104 pc
          l_c     = 180.4 degrees
          l_omega = -296.1 degrees
          phi     = 17.2 degrees

      Coordinate interpretation:

          x, y : Galactic XY plane
          z    : vertical Galactic coordinate

      The ellipse is first rotated in the XY plane by l_omega,
      then tilted about the X axis by phi.
    */
    const a = 373.0;
    const b = 233.0;

    const dC = 104.0;
    const lC = THREE.MathUtils.degToRad(180.4);

    const lOmega = THREE.MathUtils.degToRad(-296.1);
    const phi = THREE.MathUtils.degToRad(17.2);

    /*
      More points make the line smoother.

      500 is already very smooth. The final point duplicates the first
      so that the curve is explicitly closed.
    */
    const numberOfPoints = 500;

    const positions = new Float32Array(
        (numberOfPoints + 1) * 3
    );

    const cosOmega = Math.cos(lOmega);
    const sinOmega = Math.sin(lOmega);

    const cosPhi = Math.cos(phi);
    const sinPhi = Math.sin(phi);

    const xCentre = dC * Math.cos(lC);
    const yCentre = dC * Math.sin(lC);

    for (let i = 0; i <= numberOfPoints; i++) {
        const theta =
            (i / numberOfPoints) * Math.PI * 2.0;

        /*
          Unrotated ellipse in its own XY plane.
        */
        const x0 = a * Math.cos(theta);
        const y0 = b * Math.sin(theta);
        const z0 = 0.0;

        /*
          First rotation: around Galactic Z.

          This is equivalent to your Python section:

              x_l = x*cos(l_omega) - y*sin(l_omega)
              y_l = x*sin(l_omega) + y*cos(l_omega)
        */
        const x1 = x0 * cosOmega - y0 * sinOmega;
        const y1 = x0 * sinOmega + y0 * cosOmega;
        const z1 = z0;

        /*
          Second rotation: tilt around Galactic X.

          This is equivalent to your Python section:

              y_phi = y*cos(phi) - z*sin(phi)
              z_phi = y*sin(phi) + z*cos(phi)

          In your 2-D plotting code, z_phi was calculated but not
          returned. Here we retain it.
        */
        const x2 = x1;
        const y2 = y1 * cosPhi - z1 * sinPhi;
        const z2 = y1 * sinPhi + z1 * cosPhi;

        /*
          Translate the ellipse centre in the Galactic XY plane.

          The original Python function has no vertical centre offset,
          so the ellipse centre remains at z = 0.
        */
        const xFinal = x2 + xCentre;
        const yFinal = y2 + yCentre;
        const zFinal = z2;

        const index = 3 * i;

        positions[index + 0] = xFinal;
        positions[index + 1] = yFinal;
        positions[index + 2] = zFinal;
    }

    const geometry = new LineGeometry();

    geometry.setPositions(positions);

    /*
      Orange Gould Belt line.

      `linewidth` is in pixels because `worldUnits: false`.
      This is normally preferable for an interactive scientific viewer:
      the line remains readable while zooming in and out.
    */
    gouldBeltMaterial = new LineMaterial({
        color: 0xff8c00,

        linewidth: params.gouldBeltLineWidth,

        transparent: true,
        opacity: params.gouldBeltOpacity,

        depthTest: true,
        depthWrite: false,

        worldUnits: false,

        /*
          Avoid colour-management changes to this explicit RGB orange.
        */
        toneMapped: false,
    });

    /*
      LineMaterial needs the current canvas resolution.
      This is updated again whenever the browser is resized.
    */
    gouldBeltMaterial.resolution.set(
        window.innerWidth,
        window.innerHeight
    );

    const line = new Line2(
        geometry,
        gouldBeltMaterial
    );

    line.name = 'Gould Belt model (Perrot & Grenier 2003)';

    return line;
}


function makeOutline(color, opacity) {
    const geometry = new THREE.EdgesGeometry(
        new THREE.BoxGeometry(1, 1, 1)
    );

    const material = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthTest: false,
        depthWrite: false,
    });

    return new THREE.LineSegments(geometry, material);
}


function stableInterval(a, b, low, high) {
    let left = THREE.MathUtils.clamp(
        finiteNumber(a, low),
        low,
        high
    );

    let right = THREE.MathUtils.clamp(
        finiteNumber(b, high),
        low,
        high
    );

    if (left > right) {
        [left, right] = [right, left];
    }

    const epsilon = Math.max((high - low) * 1.0e-3, 1.0e-6);

    if (right - left < epsilon) {
        if (right + epsilon <= high) {
            right += epsilon;
        } else {
            left -= epsilon;
        }
    }

    return [left, right];
}


function readRange(value, name) {
    if (!Array.isArray(value) || value.length !== 2) {
        throw new Error(`Invalid ${name} in density.json.`);
    }

    const low = Number(value[0]);
    const high = Number(value[1]);

    if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) {
        throw new Error(`Invalid numerical range for ${name}.`);
    }

    return [low, high];
}


function finiteNumber(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
}


async function loadJSON(url) {
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Could not load ${url}: HTTP ${response.status}`);
    }

    return response.json();
}


async function loadArrayBuffer(url) {
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Could not load ${url}: HTTP ${response.status}`);
    }

    return response.arrayBuffer();
}


function onResize() {
    renderer.setSize(window.innerWidth, window.innerHeight);

    if (camera) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
    }

    /*
      Required by THREE.LineMaterial / Line2.

      Without this, the Gould Belt line may have an incorrect apparent
      thickness after resizing the browser window.
    */
    if (gouldBeltMaterial) {
        gouldBeltMaterial.resolution.set(
            window.innerWidth,
            window.innerHeight
        );
    }

    requestRender();
}


function reportError(error) {
    console.error(error);

    const message = error instanceof Error
        ? error.message
        : String(error);

    status.textContent = 'The volume viewer could not start.';

    errorBox.hidden = false;
    errorBox.textContent =
        `Unable to start the Three.js volume viewer.\n\n`
        + `${message}\n\n`
        + `Open this project through a local HTTP server, not by double-clicking `
        + `index.html with file:///.`;
}
