import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19.2/dist/lil-gui.esm.min.js';


const MAX_STEPS = 512;

const CLUSTER_GROUP_COLOURS = Object.freeze({
    alphaPer: 0xff00ff,  // magenta
    cr135: 0xff8c00,     // orange
    gammaVel: 0xdc143c,  // crimson
    m6: 0x00ffff,        // aqua
    other: 0x9e9e9e,     // gray
});


function getClusterGroupColour(group) {
    return (
        CLUSTER_GROUP_COLOURS[group.id]
        ?? group.color
        ?? 0xffffff
    );
}


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
const timeControl = document.getElementById('time-control');
const timeSlider = document.getElementById('time-slider');
const timeReadout = document.getElementById('time-readout');
const timeMinimum = document.getElementById('time-minimum');
const timeMaximum = document.getElementById('time-maximum');
const timeTicks = document.getElementById('time-ticks');

const renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: 'high-performance',
});

renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x02050a, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;

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

/*
  Contains the trajectory data, point geometry, material, attributes,
  and current trajectory frame.
*/
let clusterLayer = null;

let timeSliderInitialised = false;

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
    const [
        metadata,
        rawBuffer,
        colorMap,
        clusterMetadata,
        clusterRawBuffer,
    ] = await Promise.all([
        loadJSON('./data/density.json'),
        loadArrayBuffer('./data/density.u8'),
        new THREE.TextureLoader().loadAsync('./data/freeze.png'),

        loadJSON('./data/cluster_trajectories.json'),
        loadArrayBuffer('./data/cluster_trajectories.f32'),
    ]);


    /*
      Validate and decode the cluster metadata and binary position array.
    */
    const clusterData = parseClusterDataset(
        clusterMetadata,
        clusterRawBuffer
    );


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
    gouldBeltLineWidth: 5.5,

    /*
      0 = fully transparent
      1 = fully opaque
    */
    gouldBeltOpacity: 0.95,

    /*
      ----------------------------------------------------------------
      Stellar cluster trajectory controls
      ----------------------------------------------------------------
    */
    showClusters: Boolean(
        clusterData.defaultControls.visible ?? true
    ),

    clusterTime: clusterData.timesMyr[
        clusterData.zeroTimeIndex
    ],

    colorClustersByGroup: Boolean(
        clusterData.defaultControls.colorByGroup ?? true
    ),

    clusterMinSize: finiteNumber(
        clusterData.defaultControls.minMarkerSize,
        9.0
    ),

    clusterMaxSize: finiteNumber(
        clusterData.defaultControls.maxMarkerSize,
        22.0
    ),


    resetCrop: () => {},
    resetView: () => {},
    resetClusterTime: () => {},
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


    /*
      Create the dynamic point layer containing all clusters.
      The helper function adds the points to the Three.js scene.
    */
    clusterLayer = createClusterLayer(clusterData);

    /*
      Start at t = 0 Myr.
    */
    setClusterFrameFromTime(
        params.clusterTime
    );

    /*
      Set marker radii and colours.
    */
    updateClusterStyle(true);

    /*
      Create the lower-centered trajectory slider.
    */
    initialiseExternalTimeSlider();

    createGUI();

    controls.addEventListener('change', requestRender);

    const volumeMiB = rawBuffer.byteLength / 1024**2;

    const clusterTimeMin = clusterData.timesMyr[0];
    const clusterTimeMax = clusterData.timesMyr[
        clusterData.timesMyr.length - 1
    ];

    status.textContent =
        `${nx} × ${ny} × ${nz} density texture · `
        + `${clusterData.clusters.length} clusters · `
        + `${clusterData.timesMyr.length} trajectory epochs `
        + `(${clusterTimeMin} to ${clusterTimeMax} Myr)`;

    requestRender();
}


/* -------------------------------------------------------------------------- */
/* GUI                                                                        */
/* -------------------------------------------------------------------------- */

function createGUI() {
    const gui = new GUI({
        title: '',
        width: 340,
    });

    gui.domElement.style.zIndex = '20';

    const watched = (controller) => {
        controller.onChange(() => {
            requestRender();
        });

        return controller;
    };


    /* ====================================================================== */
    /* OB STAR DENSITY FIELD                                                  */
    /* ====================================================================== */

    const densityFolder = gui.addFolder(
        'OB star density field [Pantaleoni et al. 2025]'
    );

    /*
      The volume is only rendered at t = 0 Myr.
      It is automatically hidden at all other time values.
    */
    watched(
        densityFolder
            .add(params, 'showDensity')
            .name('Visible at t = 0')
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


    const guideFolder = densityFolder.addFolder(
        'Volume guides'
    );

    watched(
        guideFolder
            .add(params, 'showBounds')
            .name('Show domain box')
    );

    watched(
        guideFolder
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


    /* ====================================================================== */
    /* GOULD BELT MODEL                                                       */
    /* ====================================================================== */

    const gouldFolder = gui.addFolder(
        "Gould's Belt model [Perrot & Grenier 2003]"
    );

    /*
      Like the density field, the Gould Belt model is displayed only at
      t = 0 Myr. It is automatically hidden at other trajectory times.
    */
    watched(
        gouldFolder
            .add(params, 'showGouldBelt')
            .name('Visible at t = 0')
    );

    watched(
        gouldFolder
            .add(
                params,
                'gouldBeltLineWidth',
                1.0,
                15.0,
                0.25
            )
            .name('Line width [px]')
    );

    watched(
        gouldFolder
            .add(
                params,
                'gouldBeltOpacity',
                0.0,
                1.0,
                0.01
            )
            .name('Opacity')
    );


    /* ====================================================================== */
    /* STELLAR CLUSTER TRAJECTORIES                                           */
    /* ====================================================================== */

    const clusterFolder = gui.addFolder(
        'Young stellar clusters [Hunt & Reffert 2023]'
    );

    watched(
        clusterFolder
            .add(params, 'showClusters')
            .name('Visible')
    );


    const markerFolder = clusterFolder.addFolder(
        'Marker appearance'
    );

    watched(
        markerFolder
            .add(params, 'colorClustersByGroup')
            .name('Color by group')
    );

    watched(
        markerFolder
            .add(
                params,
                'clusterMinSize',
                1.0,
                80.0,
                0.5
            )
            .name('Min sphere diameter [px]')
    );

    watched(
        markerFolder
            .add(
                params,
                'clusterMaxSize',
                1.0,
                120.0,
                0.5
            )
            .name('Max sphere diameter [px]')
    );

    markerFolder.open();


    params.resetClusterTime = () => {
        params.clusterTime = clusterLayer.timesMyr[
            clusterLayer.zeroTimeIndex
        ];

        setClusterFrameFromTime(
            params.clusterTime
        );

        updateExternalTimeSlider();

        requestRender();
    };

    clusterFolder
        .add(params, 'resetClusterTime')
        .name('Go to t = 0 Myr');


    /*
      Initial menu state.

      There are only three top-level scientific-layer menus.
      Change .open() / .close() if you prefer a different startup state.
    */
    densityFolder.close();
    gouldFolder.close();
    clusterFolder.close();
}


/* -------------------------------------------------------------------------- */
/* UPDATE UNIFORMS                                                            */
/* -------------------------------------------------------------------------- */

function syncUniforms() {
    /*
      Update cluster positions for the selected trajectory epoch.
    */
    const clusterFrameIndex = setClusterFrameFromTime(
        params.clusterTime
    );

    /*
      Recalculate marker colours and sizes only if a relevant control
      actually changed.
    */
    updateClusterStyle();

    /*
      The density field and Gould Belt model are physically defined for
      the present-day frame, t = 0 Myr.

      They are intentionally hidden at all non-zero trajectory epochs.
    */
    const atPresentDay =
        Math.abs(
            clusterLayer.timesMyr[clusterFrameIndex]
        ) < 1.0e-8;

    const densityIsVisible =
        Boolean(params.showDensity)
        && atPresentDay;

    const gouldBeltIsVisible =
        Boolean(params.showGouldBelt)
        && atPresentDay;


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
     Update Gould Belt line appearance.
   */
   gouldBeltMaterial.linewidth = THREE.MathUtils.clamp(
       finiteNumber(params.gouldBeltLineWidth, 5.5),
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
      Cluster layer visibility is independent of the time epoch.
    */
    for (const groupLayer of clusterLayer.groupLayers) {
        groupLayer.mesh.visible = Boolean(
            params.showClusters
        );
    }

    /*
      Density field is available only at t = 0 Myr.
    */
    volumeMesh.visible = densityIsVisible;

    /*
      Gould Belt model is available only at t = 0 Myr.
    */
    gouldBeltLine.visible = gouldBeltIsVisible;

    /*
      Keep Gould Belt visual controls working.
    */
    gouldBeltMaterial.linewidth = THREE.MathUtils.clamp(
        finiteNumber(params.gouldBeltLineWidth, 5.5),
        0.5,
        30.0
    );

    gouldBeltMaterial.opacity = THREE.MathUtils.clamp(
        finiteNumber(params.gouldBeltOpacity, 0.95),
        0.0,
        1.0
    );

    /*
      The volume guide boxes belong to the density-field layer.
    */
    outerOutline.visible =
        densityIsVisible
        && Boolean(params.showBounds);

    clipOutline.visible =
        densityIsVisible
        && Boolean(params.showCropBox);
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

/* -------------------------------------------------------------------------- */
/* EXTERNAL TIME SLIDER                                                       */
/* -------------------------------------------------------------------------- */

function formatMyr(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return '0';
    }

    /*
      Most of your values are integral Myr values, but this also supports
      non-integer time grids if you ever need them later.
    */
    if (Math.abs(number - Math.round(number)) < 1.0e-8) {
        return String(Math.round(number));
    }

    return number.toFixed(2);
}


function updateExternalTimeSlider() {
    if (!clusterLayer || !timeSlider) {
        return;
    }

    const firstTime = clusterLayer.timesMyr[0];

    const lastTime = clusterLayer.timesMyr[
        clusterLayer.timesMyr.length - 1
    ];

    const currentTime = clusterLayer.timesMyr[
        nearestClusterFrameIndex(params.clusterTime)
    ];

    const denominator = lastTime - firstTime;

    const fraction = denominator > 0.0
        ? THREE.MathUtils.clamp(
            (currentTime - firstTime) / denominator,
            0.0,
            1.0
        )
        : 0.5;

    timeSlider.value = String(currentTime);

    timeReadout.textContent =
        `t = ${formatMyr(currentTime)} Myr`;

    /*
      Move the floating text so that it follows the thumb.
    */
    timeReadout.style.left =
        `${100.0 * fraction}%`;

    timeSlider.setAttribute(
        'aria-valuetext',
        `t = ${formatMyr(currentTime)} Myr`
    );
}

function createTimeTicks() {
    if (!clusterLayer || !timeTicks) {
        return;
    }

    const firstTime = clusterLayer.timesMyr[0];

    const lastTime = clusterLayer.timesMyr[
        clusterLayer.timesMyr.length - 1
    ];

    const range = lastTime - firstTime;

    if (range <= 0) {
        return;
    }

    /*
      Clear old ticks if this function is called again.
    */
    timeTicks.replaceChildren();

    /*
      Add ticks every 10 Myr.

      For a -61 ... +61 Myr slider this creates marks at:
      -60, -50, ..., -10, 0, +10, ..., +60 Myr.
    */
    for (let time = -60; time <= 60; time += 10) {
        if (time < firstTime || time > lastTime) {
            continue;
        }

        const fraction =
            (time - firstTime) / range;

        const tick = document.createElement('span');

        tick.className =
            time === 0
                ? 'time-tick now'
                : 'time-tick';

        tick.style.left = `${100.0 * fraction}%`;

        /*
          Useful browser tooltip when hovering a tick mark.
        */
        tick.title =
            time === 0
                ? 'Now: t = 0 Myr'
                : `t = ${time} Myr`;

        timeTicks.appendChild(tick);
    }
}

function initialiseExternalTimeSlider() {
    if (
        !clusterLayer
        || !timeControl
        || !timeSlider
        || timeSliderInitialised
    ) {
        return;
    }

    const firstTime = clusterLayer.timesMyr[0];

    const lastTime = clusterLayer.timesMyr[
        clusterLayer.timesMyr.length - 1
    ];

    timeSlider.min = String(firstTime);
    timeSlider.max = String(lastTime);
    timeSlider.step = String(
        clusterLayer.timeStepMyr
    );

    timeMinimum.textContent = 'Past';
    timeMaximum.textContent = 'Future';

    /*
      Create the fixed 10-Myr tick marks, including the larger t = 0 mark.
    */
    createTimeTicks();

    /*
      While the user drags the thumb, immediately update cluster positions.
    */
    timeSlider.addEventListener('input', () => {
        const requestedTime = Number(
            timeSlider.value
        );

        const frameIndex = nearestClusterFrameIndex(
            requestedTime
        );

        params.clusterTime = clusterLayer.timesMyr[
            frameIndex
        ];

        setClusterFrameFromTime(
            params.clusterTime
        );

        updateExternalTimeSlider();

        requestRender();
    });

    timeSliderInitialised = true;

    timeControl.hidden = false;

    updateExternalTimeSlider();
}

/* -------------------------------------------------------------------------- */
/* CLUSTER TRAJECTORY DATA                                                     */
/* -------------------------------------------------------------------------- */

function parseClusterDataset(metadata, rawBuffer) {
    if (
        !metadata
        || !Array.isArray(metadata.timesMyr)
        || !Array.isArray(metadata.groups)
        || !Array.isArray(metadata.clusters)
    ) {
        throw new Error(
            'cluster_trajectories.json has an invalid structure.'
        );
    }

    const timesMyr = metadata.timesMyr.map(Number);

    if (
        timesMyr.length < 2
        || !timesMyr.every(Number.isFinite)
    ) {
        throw new Error(
            'cluster_trajectories.json contains an invalid time grid.'
        );
    }

    for (let index = 1; index < timesMyr.length; index++) {
        if (timesMyr[index] <= timesMyr[index - 1]) {
            throw new Error(
                'Cluster trajectory times must be strictly increasing.'
            );
        }
    }

    const timeStepMyr = timesMyr[1] - timesMyr[0];

    for (let index = 2; index < timesMyr.length; index++) {
        const currentStep =
            timesMyr[index] - timesMyr[index - 1];

        if (Math.abs(currentStep - timeStepMyr) > 1.0e-7) {
            throw new Error(
                'Cluster trajectory time samples are not uniformly spaced.'
            );
        }
    }

    const zeroTimeIndex = timesMyr.findIndex(
        (time) => Math.abs(time) < 1.0e-8
    );

    if (zeroTimeIndex < 0) {
        throw new Error(
            'Cluster trajectories do not contain t = 0 Myr.'
        );
    }

    const groups = metadata.groups.map((group, index) => {
        if (!group || typeof group !== 'object') {
            throw new Error(
                `Invalid cluster group at index ${index}.`
            );
        }

        return {
            id: String(group.id ?? `group-${index}`),
            label: String(group.label ?? `Group ${index}`),
            color: String(group.color ?? '#ffffff'),
        };
    });

    const clusters = metadata.clusters.map((cluster, index) => {
        if (!cluster || typeof cluster !== 'object') {
            throw new Error(
                `Invalid cluster record at index ${index}.`
            );
        }

        const groupIndex = Number(cluster.groupIndex);
        const nStars = Number(cluster.nStars);

        if (
            !Number.isInteger(groupIndex)
            || groupIndex < 0
            || groupIndex >= groups.length
        ) {
            throw new Error(
                `Cluster '${cluster.name}' has an invalid groupIndex.`
            );
        }

        if (!Number.isFinite(nStars)) {
            throw new Error(
                `Cluster '${cluster.name}' has an invalid nStars value.`
            );
        }

        return {
            name: String(cluster.name),
            nStars,
            groupIndex,
        };
    });

    if (clusters.length === 0) {
        throw new Error(
            'No clusters were found in cluster_trajectories.json.'
        );
    }

    const expectedFloatCount =
        timesMyr.length
        * clusters.length
        * 3;

    const expectedByteLength =
        expectedFloatCount
        * Float32Array.BYTES_PER_ELEMENT;

    if (rawBuffer.byteLength !== expectedByteLength) {
        throw new Error(
            'cluster_trajectories.f32 has an unexpected size.\n\n'
            + `Expected: ${expectedByteLength} bytes\n`
            + `Found:    ${rawBuffer.byteLength} bytes`
        );
    }

    const trajectory = new Float32Array(rawBuffer);

    for (let index = 0; index < trajectory.length; index++) {
        if (!Number.isFinite(trajectory[index])) {
            throw new Error(
                'cluster_trajectories.f32 contains invalid numerical values.'
            );
        }
    }

    return {
        timesMyr,
        timeStepMyr,
        zeroTimeIndex,

        groups,
        clusters,

        trajectory,

        defaultControls: metadata.defaultControls ?? {},
    };
}


function updateClusterTransforms(force = false) {
    if (!clusterLayer) {
        return;
    }

    const frameIndex = clusterLayer.frameIndex;

    if (frameIndex < 0) {
        return;
    }

    const numberOfClusters = clusterLayer.clusters.length;

    const valuesPerFrame =
        numberOfClusters * 3;

    const positionOffset =
        frameIndex * valuesPerFrame;

    const trajectory = clusterLayer.trajectory;
    const radiiPc = clusterLayer.radiiPc;

    const dummy = clusterLayer.dummy;

    /*
      Update every group mesh independently.

      `clusterIndex` is the index in the complete catalogue.
      `localIndex` is the instance index within that specific group mesh.
    */
    for (const groupLayer of clusterLayer.groupLayers) {
        const {
            mesh,
            clusterIndices,
        } = groupLayer;

        for (
            let localIndex = 0;
            localIndex < clusterIndices.length;
            localIndex++
        ) {
            const clusterIndex =
                clusterIndices[localIndex];

            const coordinateIndex =
                positionOffset + 3 * clusterIndex;

            const x = trajectory[coordinateIndex + 0];
            const y = trajectory[coordinateIndex + 1];
            const z = trajectory[coordinateIndex + 2];

            const radius = Math.max(
                radiiPc[clusterIndex],
                0.001
            );

            dummy.position.set(x, y, z);

            /*
              SphereGeometry radius is 1, so this gives the actual
              physical radius in pc.
            */
            dummy.scale.set(radius, radius, radius);

            dummy.updateMatrix();

            mesh.setMatrixAt(
                localIndex,
                dummy.matrix
            );
        }

        mesh.instanceMatrix.needsUpdate = true;
    }
}


/* -------------------------------------------------------------------------- */
/* CREATE THE GPU POINT LAYER                                                 */
/* -------------------------------------------------------------------------- */

function createClusterLayer(dataset) {
    const numberOfClusters = dataset.clusters.length;

    /*
      A single unit sphere geometry is shared by all five groups.

      Each individual cluster receives its physical radius through an
      instance transformation matrix.
    */
    const sphereGeometry = new THREE.SphereGeometry(
        1.0,
        16,
        12
    );

    /*
      Create one InstancedMesh for each cluster group.

      This avoids the per-instance GPU colour-buffer issue entirely.
      Each group has a normal material with a direct colour.
    */
    const groupLayers = dataset.groups
        .map((group, groupIndex) => {
            const clusterIndices = [];

            for (
                let clusterIndex = 0;
                clusterIndex < dataset.clusters.length;
                clusterIndex++
            ) {
                if (
                    dataset.clusters[clusterIndex].groupIndex
                    === groupIndex
                ) {
                    clusterIndices.push(clusterIndex);
                }
            }

            /*
              A group may theoretically be empty. Do not create a mesh
              in that case.
            */
            if (clusterIndices.length === 0) {
                return null;
            }

            const baseColour = new THREE.Color(
                getClusterGroupColour(group)
            );

            const material = new THREE.MeshPhongMaterial({
                /*
                  This is the visible group colour.
                */
                color: baseColour,

                /*
                  A faint emissive component ensures that the sphere is
                  recognisably coloured even on its dark/shadowed side.
                */
                emissive: baseColour.clone().multiplyScalar(0.12),
                emissiveIntensity: 1.0,

                shininess: 55,
                specular: 0x666666,

                /*
                  The density field is transparent and must be rendered
                  first. Keeping spheres in the transparent pass with
                  opacity = 1 draws them afterward.

                  They remain visually opaque.
                */
                transparent: true,
                opacity: 1.0,

                /*
                  Important: real depth-buffer writes fix the overlap
                  problem between nearby and distant spheres.
                */
                depthTest: true,
                depthWrite: true,

                toneMapped: false,
            });

            const mesh = new THREE.InstancedMesh(
                sphereGeometry,
                material,
                clusterIndices.length
            );

            mesh.name = group.label;

            mesh.instanceMatrix.setUsage(
                THREE.DynamicDrawUsage
            );

            /*
              Clusters may move outside the initial bounding area during
              traceback/forward integration.
            */
            mesh.frustumCulled = false;

            /*
              Draw after volume and Gould Belt.
            */
            mesh.renderOrder = 10;

            scene.add(mesh);

            return {
                groupIndex,
                group,
                clusterIndices,
                mesh,
                material,
            };
        })
        .filter((groupLayer) => groupLayer !== null);


    /*
      Lighting applies only to ordinary Three.js materials such as the
      cluster spheres. It does not affect the custom volume shader.
    */
    const ambientLight = new THREE.AmbientLight(
        0xffffff,
        0.85
    );

    scene.add(ambientLight);


    const hemisphereLight = new THREE.HemisphereLight(
        0xdceeff,
        0x263041,
        0.70
    );

    scene.add(hemisphereLight);


    const directionalLight = new THREE.DirectionalLight(
        0xffffff,
        1.10
    );

    directionalLight.position.set(
        -1.0,
        1.5,
        2.0
    );

    scene.add(directionalLight);


    const nStarsValues = dataset.clusters.map(
        (cluster) => cluster.nStars
    );

    const dummy = new THREE.Object3D();

    return {
        timesMyr: dataset.timesMyr,
        timeStepMyr: dataset.timeStepMyr,
        zeroTimeIndex: dataset.zeroTimeIndex,

        groups: dataset.groups,
        clusters: dataset.clusters,
        trajectory: dataset.trajectory,

        /*
          There are now five sphere meshes rather than one.
        */
        groupLayers,

        sphereGeometry,
        dummy,

        radiiPc: new Float32Array(numberOfClusters),

        nStarsMin: Math.min(...nStarsValues),
        nStarsMax: Math.max(...nStarsValues),

        referenceDistance: camera.position.distanceTo(
            controls.target
        ),

        frameIndex: -1,
        styleSignature: '',
    };
}


/* -------------------------------------------------------------------------- */
/* TIME SELECTION                                                             */
/* -------------------------------------------------------------------------- */

function nearestClusterFrameIndex(timeMyr) {
    if (!clusterLayer) {
        return 0;
    }

    const fallbackTime = clusterLayer.timesMyr[
        clusterLayer.zeroTimeIndex
    ];

    const requestedTime = finiteNumber(
        timeMyr,
        fallbackTime
    );

    let bestIndex = 0;

    let bestDistance = Math.abs(
        requestedTime - clusterLayer.timesMyr[0]
    );

    for (
        let index = 1;
        index < clusterLayer.timesMyr.length;
        index++
    ) {
        const distance = Math.abs(
            requestedTime - clusterLayer.timesMyr[index]
        );

        if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = index;
        }
    }

    return bestIndex;
}


function setClusterFrameFromTime(timeMyr) {
    if (!clusterLayer) {
        return -1;
    }

    const frameIndex = nearestClusterFrameIndex(
        timeMyr
    );

    const canonicalTime = clusterLayer.timesMyr[
        frameIndex
    ];

    /*
      Keep the parameter exactly on one of the available precomputed
      trajectory epochs.
    */
    params.clusterTime = canonicalTime;

    if (clusterLayer.frameIndex === frameIndex) {
        return frameIndex;
    }

    clusterLayer.frameIndex = frameIndex;

    /*
      The position of every sphere is encoded in its instance matrix.
    */
    updateClusterTransforms(true);

    return frameIndex;
}


/* -------------------------------------------------------------------------- */
/* MARKER STYLE                                                               */
/* -------------------------------------------------------------------------- */

function updateClusterStyle(force = false) {
    if (!clusterLayer) {
        return;
    }

    const minDiameterPx = THREE.MathUtils.clamp(
        finiteNumber(params.clusterMinSize, 8.0),
        1.0,
        120.0
    );

    const maxDiameterPx = THREE.MathUtils.clamp(
        finiteNumber(params.clusterMaxSize, 60.0),
        1.0,
        200.0
    );

    const colorByGroup = Boolean(
        params.colorClustersByGroup
    );

    const viewportHeight = Math.max(
        renderer.domElement.clientHeight,
        1
    );

    const styleSignature = [
        minDiameterPx,
        maxDiameterPx,
        colorByGroup,
        viewportHeight,
    ].join('|');

    if (
        !force
        && clusterLayer.styleSignature === styleSignature
    ) {
        return;
    }

    const nStarsMin = clusterLayer.nStarsMin;
    const nStarsMax = clusterLayer.nStarsMax;

    const nStarsRange =
        nStarsMax - nStarsMin;

    const verticalFovRadians = THREE.MathUtils.degToRad(
        camera.fov
    );

    /*
      Convert the old screen-pixel marker diameter into a real physical
      sphere radius, in pc, at the initial reference camera distance.
    */
    const pixelsToPhysicalRadius =
        clusterLayer.referenceDistance
        * Math.tan(verticalFovRadians * 0.5)
        / viewportHeight;

    /*
      Update the physical radius of every cluster.
    */
    for (
        let clusterIndex = 0;
        clusterIndex < clusterLayer.clusters.length;
        clusterIndex++
    ) {
        const cluster = clusterLayer.clusters[
            clusterIndex
        ];

        const normalizedNStars =
            nStarsRange > 0.0
                ? THREE.MathUtils.clamp(
                    (cluster.nStars - nStarsMin)
                    / nStarsRange,
                    0.0,
                    1.0
                )
                : 0.0;

        /*
          Same parabolic relation as your original Python marker-size
          prescription.
        */
        const markerDiameterPx =
            minDiameterPx
            + (maxDiameterPx - minDiameterPx)
            * normalizedNStars
            * normalizedNStars;

        const radiusPc =
            markerDiameterPx
            * pixelsToPhysicalRadius;

        clusterLayer.radiiPc[clusterIndex] = Math.max(
            radiusPc,
            0.25
        );
    }

    /*
      Update the colour of each of the five group materials.

      When colour-by-group is disabled, every group material becomes white.
    */
    for (const groupLayer of clusterLayer.groupLayers) {
        const displayColour = new THREE.Color(
            colorByGroup
                ? getClusterGroupColour(groupLayer.group)
                : 0xffffff
        );

        groupLayer.material.color.copy(
            displayColour
        );

        /*
          The emissive contribution is deliberately small: it preserves
          recognisable colour in darkness while retaining sphere shading.
        */
        groupLayer.material.emissive
            .copy(displayColour)
            .multiplyScalar(0.12);

        groupLayer.material.needsUpdate = true;
    }

    /*
      Radii changed, so update the instance transform matrices.
    */
    updateClusterTransforms(true);

    clusterLayer.styleSignature = styleSignature;
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
    renderer.setSize(
        window.innerWidth,
        window.innerHeight
    );

    if (camera) {
        camera.aspect =
            window.innerWidth / window.innerHeight;

        camera.updateProjectionMatrix();
    }

    if (gouldBeltMaterial) {
        gouldBeltMaterial.resolution.set(
            window.innerWidth,
            window.innerHeight
        );
    }

    /*
      Sphere marker radii are calibrated from the reference screen-pixel
      sizes, so update them after resizing the browser window.
    */
    if (clusterLayer) {
        updateClusterStyle(true);
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
