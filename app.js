import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19.2/dist/lil-gui.esm.min.js';


const MAX_STEPS = 512;

/*
  A cluster begins fading once we go farther into the past than its age.

  Example:
      age = 20 Myr

      t = -20 Myr  -> opacity 1.0
      t = -22.5 Myr -> opacity 0.5
      t = -25 Myr  -> opacity 0.0
*/
const CLUSTER_BIRTH_FADE_MYR = 5.0;


/*
  The present-day OB density field and Gould Belt model are gradually
  faded away when moving away from t = 0.

  They reach zero opacity at t = -5 Myr and t = +5 Myr.
*/

/*
  OB-star density field fades from its nominal opacity at t = 0
  to zero opacity at |t| = 5 Myr.
*/
const OB_STAR_FIELD_FADE_MYR = 5.0;


/*
  Gould Belt model fades more rapidly: from nominal opacity at t = 0
  to zero opacity at |t| = 3 Myr.
*/
const GOULD_BELT_FADE_MYR = 3.0;


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

    steps: Math.round(finiteNumber(defaults.steps, 100)),

    showBounds: true,

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
    gouldBeltOpacity: 0.50,

    /*
      ----------------------------------------------------------------
      Stellar cluster trajectory controls
      ----------------------------------------------------------------
    */
    showClusters: Boolean(
        clusterData.defaultControls.visible ?? true
    ),

    /*
      Trajectory trail controls.

      Trails are visible by default, but at t = 0 there is no trail length,
      so nothing is drawn until the time slider moves into the past/future.
    */
    showTrails: true,

    trailLineWidth: 2.0,
    trailOpacity: 0.75,

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
      The domain box is added directly to the scene, not as a child of the
      density volume. Therefore it remains available at all times, even when
      the OB-star field itself has faded out.
    */
    outerOutline = makeOutline(0x6481a0, 0.38);

    outerOutline.position.copy(centre);
    outerOutline.scale.copy(extent);

    outerOutline.renderOrder = 2;

    scene.add(outerOutline);


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
      Set sphere radii and group colours.
    */
    updateClusterStyle(true);

    /*
      Set initial line width, opacity, and colours for all trails.
    */
    updateClusterTrailStyle(true);

    /*
      Create the lower-centered trajectory slider.
    */
    initialiseExternalTimeSlider();

    createGUI();

    controls.addEventListener('change', requestRender);

    /*
      The HUD intentionally contains only the title and interaction hints.
    */
    status.textContent = '';

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
    /* 1. OB STAR DENSITY FIELD                                               */
    /* ====================================================================== */

    const densityFolder = gui.addFolder(
        'OB star density field [Pantaleoni et al. 2025]'
    );

    watched(
        densityFolder
            .add(params, 'showDensity')
            .name('Show OB stars')
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

    const qualityFolder = densityFolder.addFolder(
        'Render quality'
    );

    watched(
        qualityFolder
            .add(params, 'steps', 32, MAX_STEPS, 1)
            .name('Ray-march sampling')
    );


    /* ====================================================================== */
    /* 2. YOUNG STELLAR CLUSTERS                                              */
    /* ====================================================================== */

    const clusterFolder = gui.addFolder(
        'Young stellar clusters [Hunt & Reffert 2023]'
    );

    watched(
        clusterFolder
            .add(params, 'showClusters')
            .name('Show clusters')
    );

    watched(
        clusterFolder
            .add(params, 'colorClustersByGroup')
            .name('Colour by cluster families')
    );

    watched(
        clusterFolder
            .add(params, 'showTrails')
            .name('Show cluster trails')
    );

    watched(
        clusterFolder
            .add(
                params,
                'trailLineWidth',
                0.5,
                12.0,
                0.25
            )
            .name('Trail width [px]')
    );

    watched(
        clusterFolder
            .add(
                params,
                'trailOpacity',
                0.0,
                1.0,
                0.01
            )
            .name('Trail opacity')
    );

    const markerFolder = clusterFolder.addFolder(
        'Marker appearance'
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
            .name('Min marker size [px]')
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
            .name('Max marker size [px]')
    );


    /* ====================================================================== */
    /* 3. GOULD BELT MODEL                                                    */
    /* ====================================================================== */

    const gouldFolder = gui.addFolder(
        "Gould's Belt model [Perrot & Grenier 2003]"
    );

    watched(
        gouldFolder
            .add(params, 'showGouldBelt')
            .name('Show Gould Belt model')
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
    /* 4. GRID AND MEASURES                                                   */
    /* ====================================================================== */

    const gridFolder = gui.addFolder(
        'Grid and measures'
    );

    watched(
        gridFolder
            .add(params, 'showBounds')
            .name('Show domain box')
    );


    /*
      Keep every top-level scientific menu collapsed at startup.
    */
    densityFolder.close();
    clusterFolder.close();
    gouldFolder.close();
    gridFolder.close();
}

function timeFadeOpacity(
    timeMyr,
    fadeDurationMyr
) {
    /*
      Distance from the present epoch.
    */
    const distanceFromNow = Math.abs(timeMyr);

    /*
      Convert to [0, 1]:

          0 at t = 0
          1 at |t| >= fadeDurationMyr
    */
    const normalizedDistance = THREE.MathUtils.clamp(
        distanceFromNow / Math.max(fadeDurationMyr, 1.0e-6),
        0.0,
        1.0
    );

    /*
      Smoothstep:

          smoothstep(0, 1, x) = 3x² - 2x³

      Invert it so:

          t = 0               -> opacity factor 1
          |t| >= fade duration -> opacity factor 0
    */
    const smoothFade =
        normalizedDistance
        * normalizedDistance
        * (
            3.0
            - 2.0 * normalizedDistance
        );

    return 1.0 - smoothFade;
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

    updateClusterTrailStyle();


    /*
      The density field and Gould Belt model are physically defined for
      the present-day frame, t = 0 Myr.

      They are intentionally hidden at all non-zero trajectory epochs.
    */

    /*
      Current selected trajectory epoch in Myr.
    */
    const selectedTimeMyr =
        clusterLayer.timesMyr[clusterFrameIndex];

    /*
      Smoothly fade present-day-only objects as the user moves away from
      t = 0 Myr.

      At |t| >= 5 Myr this becomes exactly zero.
    */
    /*
      Use separate temporal fading for the two present-day models.

      OB density:
          fully visible at t = 0
          fades out by |t| = 5 Myr

      Gould Belt:
          fully visible at t = 0
          fades out by |t| = 3 Myr
    */
    const obStarFieldFade = timeFadeOpacity(
        selectedTimeMyr,
        OB_STAR_FIELD_FADE_MYR
    );

    const gouldBeltFade = timeFadeOpacity(
        selectedTimeMyr,
        GOULD_BELT_FADE_MYR
    );

    const densityIsVisible =
        Boolean(params.showDensity)
        && obStarFieldFade > 0.001;

    const gouldBeltIsVisible =
        Boolean(params.showGouldBelt)
        && gouldBeltFade > 0.001;


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

    uniforms.uLower.value = lower;
    uniforms.uUpper.value = upper;

    uniforms.uSoftness.value = THREE.MathUtils.clamp(
        finiteNumber(params.softness, 0.035),
        0.001,
        0.5
    );

    const nominalDensityOpacity = THREE.MathUtils.clamp(
        finiteNumber(params.opacity, 1.0),
        0.0,
        1.0
    );

    /*
      Fade the density field as we move away from t = 0.
    */
    uniforms.uGlobalOpacity.value =
        nominalDensityOpacity
        * obStarFieldFade;

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

    /*
      Cluster layer visibility is independent of the time epoch.
    */
    for (const groupLayer of clusterLayer.groupLayers) {
        groupLayer.mesh.visible = Boolean(
            params.showClusters
        );
    }

    /*
      Trails are independently toggleable.

      At t = 0, no trail is shown because its length would be zero.
    */
    const trailsAreVisible =
        Boolean(params.showTrails)
        && clusterFrameIndex !== clusterLayer.zeroTimeIndex;

    for (const trail of clusterLayer.trails) {
        /*
          Main line: ordinary trajectory path.
        */
        trail.line.visible = trailsAreVisible;

        /*
          The old one-piece fade line is permanently disabled.
        */
        trail.fadeLine.visible = false;

        /*
          Every one-Myr fading segment has separate age-dependent opacity.
        */
        for (const segment of trail.fadeSegments) {
            segment.line.visible =
                trailsAreVisible
                && segment.material.opacity > 0.001;
        }
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

    const nominalGouldBeltOpacity = THREE.MathUtils.clamp(
        finiteNumber(params.gouldBeltOpacity, 0.50),
        0.0,
        1.0
    );

    /*
      Fade the present-day Gould Belt model away together with the KDE field.
    */
    gouldBeltMaterial.opacity =
        nominalGouldBeltOpacity
        * gouldBeltFade;

    /*
      The domain box is independent from the OB-density visibility and
      remains visible at every time if enabled.
    */
    outerOutline.visible = Boolean(
        params.showBounds
    );
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
        const ageMyr = Number(cluster.ageMyr);

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

        if (
            !Number.isFinite(ageMyr)
            || ageMyr < 0.0
        ) {
            throw new Error(
                `Cluster '${cluster.name}' has an invalid ageMyr value.`
            );
        }

        return {
            name: String(cluster.name),
            nStars,
            ageMyr,
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


function makeAgeAwareClusterMaterial(baseColour) {
    const material = new THREE.MeshPhongMaterial({
        /*
          The group colour is assigned directly to the group material.
        */
        color: baseColour,

        /*
          A small emissive contribution keeps group colours readable on
          the dark/shadowed hemisphere of each sphere.
        */
        emissive: baseColour.clone().multiplyScalar(0.12),
        emissiveIntensity: 1.0,

        shininess: 55,
        specular: 0x666666,

        /*
          Individual alpha values are supplied through the custom
          instanceOpacity GPU attribute.
        */
        transparent: true,
        opacity: 1.0,

        /*
          Preserve correct depth handling among the spheres.
        */
        depthTest: true,
        depthWrite: true,

        toneMapped: false,
    });

    /*
      Inject one per-instance opacity attribute into MeshPhongMaterial.

      Every group mesh has an InstancedBufferAttribute named
      `instanceOpacity`. This shader extension multiplies the usual
      material opacity by that cluster-specific value.
    */
    material.onBeforeCompile = (shader) => {
        shader.vertexShader = shader.vertexShader
            .replace(
                '#include <common>',
                `
#include <common>

attribute float instanceOpacity;

varying float vInstanceOpacity;
`
            )
            .replace(
                '#include <begin_vertex>',
                `
#include <begin_vertex>

vInstanceOpacity = instanceOpacity;
`
            );

        shader.fragmentShader = shader.fragmentShader
            .replace(
                '#include <common>',
                `
#include <common>

varying float vInstanceOpacity;
`
            )
            .replace(
                'vec4 diffuseColor = vec4( diffuse, opacity );',
                `
vec4 diffuseColor = vec4( diffuse, opacity );

diffuseColor.a *= clamp(
    vInstanceOpacity,
    0.0,
    1.0
);

/*
  Do not allow completely invisible clusters to write depth values.
*/
if (diffuseColor.a < 0.001) {
    discard;
}
`
            );
    };

    /*
      Ensures Three.js knows this material uses a custom shader variant.
    */
    material.customProgramCacheKey = () => {
        return 'cluster-age-aware-phong-v1';
    };

    return material;
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
      Base geometry shared conceptually by every cluster sphere.

      Each group receives a clone because every group requires its own
      independent `instanceOpacity` attribute buffer.
    */
    const baseSphereGeometry = new THREE.SphereGeometry(
        1.0,
        16,
        12
    );

    /*
      Create one InstancedMesh per cluster group.

      This preserves your reliable direct group-colour approach:
      alpha Per -> magenta
      Cr 135 -> orange
      gamma Vel -> crimson
      M6 -> aqua
      others -> gray
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

            if (clusterIndices.length === 0) {
                return null;
            }

            /*
              Each group needs an independent geometry because the
              instance-opacity attribute length differs by group.
            */
            const geometry = baseSphereGeometry.clone();

            const instanceOpacity = new THREE.InstancedBufferAttribute(
                new Float32Array(clusterIndices.length),
                1
            );

            instanceOpacity.array.fill(1.0);

            instanceOpacity.setUsage(
                THREE.DynamicDrawUsage
            );

            geometry.setAttribute(
                'instanceOpacity',
                instanceOpacity
            );

            const baseColour = new THREE.Color(
                getClusterGroupColour(group)
            );

            const material = makeAgeAwareClusterMaterial(
                baseColour
            );

            const mesh = new THREE.InstancedMesh(
                geometry,
                material,
                clusterIndices.length
            );

            mesh.name = group.label;

            mesh.instanceMatrix.setUsage(
                THREE.DynamicDrawUsage
            );

            /*
              Traceback positions may lie outside the current camera
              framing, so disable object-level frustum culling.
            */
            mesh.frustumCulled = false;

            /*
              Cluster spheres are rendered after the density volume.
            */
            mesh.renderOrder = 10;

            scene.add(mesh);

            return {
                groupIndex,
                group,
                clusterIndices,

                geometry,
                mesh,
                material,

                instanceOpacity,
            };
        })
        .filter((groupLayer) => groupLayer !== null);


    /*
      One Line2 object per cluster trail.

      Line2 is used rather than THREE.Line because it supports a
      consistent configurable screen-space width across browsers.
    */
    const trails = dataset.clusters.map(
        (cluster, clusterIndex) => {
            const group = dataset.groups[
                cluster.groupIndex
            ];

            /*
              ----------------------------------------------------------------
              Main trail geometry/material
              ----------------------------------------------------------------

              This line is fully visible from t = 0 back to the nominal
              cluster age, or forward to the selected future epoch.
            */
            const geometry = new LineGeometry();

            geometry.setPositions(
                new Float32Array([
                    0, 0, 0,
                    0, 0, 0,
                ])
            );

            const material = new LineMaterial({
                color: getClusterGroupColour(group),

                linewidth: params.trailLineWidth,

                transparent: true,
                opacity: params.trailOpacity,

                depthTest: true,
                depthWrite: false,

                worldUnits: false,
                toneMapped: false,
            });

            material.resolution.set(
                window.innerWidth,
                window.innerHeight
            );

            const line = new Line2(
                geometry,
                material
            );

            line.name = `${cluster.name} trajectory`;

            line.renderOrder = 7;
            line.frustumCulled = false;
            line.visible = false;

            scene.add(line);


            /*
              ----------------------------------------------------------------
              Fade-tail geometry/material
              ----------------------------------------------------------------

              This second line represents only the final 5 Myr interval
              after the cluster age is exceeded.

              Its opacity smoothly drops from the nominal trail opacity to
              zero as the slider proceeds farther into the past.
            */
            const fadeGeometry = new LineGeometry();

            fadeGeometry.setPositions(
                new Float32Array([
                    0, 0, 0,
                    0, 0, 0,
                ])
            );

            const fadeMaterial = new LineMaterial({
                color: getClusterGroupColour(group),

                linewidth: params.trailLineWidth,

                transparent: true,
                opacity: 0.0,

                depthTest: true,
                depthWrite: false,

                worldUnits: false,
                toneMapped: false,
            });

            fadeMaterial.resolution.set(
                window.innerWidth,
                window.innerHeight
            );

            const fadeLine = new Line2(
                fadeGeometry,
                fadeMaterial
            );

            fadeLine.name = `${cluster.name} age fade trail`;

            /*
              Render just after the main trail.
            */
            fadeLine.renderOrder = 8;
            fadeLine.frustumCulled = false;
            fadeLine.visible = false;

            scene.add(fadeLine);

            return {
                clusterIndex,

                /*
                  Main trail: from t = 0 to the nominal cluster age.
                */
                geometry,
                material,
                line,

                /*
                  Keep these old objects temporarily so existing code does not fail.
                  They will always remain hidden after the changes below.
                */
                fadeGeometry,
                fadeMaterial,
                fadeLine,

                /*
                  Individual one-Myr age-fade line segments.

                  Each segment has its own Line2 and LineMaterial, allowing opacity
                  to decrease progressively along the path rather than making the
                  entire final section equally transparent.
                */
                fadeSegments: [],
            };
        }
    );


    /*
      Lighting affects only standard Three.js objects such as spheres.
      It does not affect the custom ray-marched KDE volume shader.
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

        groupLayers,
        trails,

        dummy,

        radiiPc: new Float32Array(numberOfClusters),

        nStarsMin: Math.min(...nStarsValues),
        nStarsMax: Math.max(...nStarsValues),

        referenceDistance: camera.position.distanceTo(
            controls.target
        ),

        frameIndex: -1,

        styleSignature: '',
        trailStyleSignature: '',
    };
}


function clusterAgeOpacity(
    selectedTimeMyr,
    clusterAgeMyr
) {
    /*
      Future and present-day epochs remain fully visible.

      For a cluster of age A:
          t >= -A         -> opacity 1
          t = -A - 2.5    -> opacity 0.5
          t <= -A - 5     -> opacity 0
    */
    if (selectedTimeMyr >= -clusterAgeMyr) {
        return 1.0;
    }

    return THREE.MathUtils.clamp(
        1.0
        + (
            selectedTimeMyr
            + clusterAgeMyr
        ) / CLUSTER_BIRTH_FADE_MYR,
        0.0,
        1.0
    );
}


function updateClusterAgeOpacity(selectedTimeMyr) {
    if (!clusterLayer) {
        return;
    }

    for (const groupLayer of clusterLayer.groupLayers) {
        const {
            clusterIndices,
            instanceOpacity,
        } = groupLayer;

        for (
            let localIndex = 0;
            localIndex < clusterIndices.length;
            localIndex++
        ) {
            const clusterIndex = clusterIndices[
                localIndex
            ];

            const cluster = clusterLayer.clusters[
                clusterIndex
            ];

            instanceOpacity.array[localIndex] =
                clusterAgeOpacity(
                    selectedTimeMyr,
                    cluster.ageMyr
                );
        }

        instanceOpacity.needsUpdate = true;
    }
}


function buildTrailPositions(
    clusterIndex,
    firstFrame,
    lastFrame,
    frameDirection
) {
    const numberOfClusters =
        clusterLayer.clusters.length;

    const numberOfPoints =
        Math.abs(lastFrame - firstFrame)
        + 1;

    const positions = new Float32Array(
        numberOfPoints * 3
    );

    for (
        let pointIndex = 0;
        pointIndex < numberOfPoints;
        pointIndex++
    ) {
        const frameIndex =
            firstFrame
            + pointIndex * frameDirection;

        const sourceIndex =
            (
                frameIndex * numberOfClusters
                + clusterIndex
            ) * 3;

        const destinationIndex =
            pointIndex * 3;

        positions[destinationIndex + 0] =
            clusterLayer.trajectory[sourceIndex + 0];

        positions[destinationIndex + 1] =
            clusterLayer.trajectory[sourceIndex + 1];

        positions[destinationIndex + 2] =
            clusterLayer.trajectory[sourceIndex + 2];
    }

    return positions;
}


function replaceTrailGeometry(
    trail,
    geometryProperty,
    lineProperty,
    positions
) {
    const oldGeometry = trail[geometryProperty];

    const newGeometry = new LineGeometry();

    newGeometry.setPositions(positions);

    trail[geometryProperty] = newGeometry;
    trail[lineProperty].geometry = newGeometry;

    oldGeometry.dispose();

    trail[lineProperty].computeLineDistances();
}


function disposeFadeSegments(trail) {
    /*
      Remove and dispose every individual age-fade segment for one cluster.

      This is called each time the selected trajectory epoch changes, so
      the visible fade region always exactly matches the selected time.
    */
    for (const segment of trail.fadeSegments) {
        scene.remove(segment.line);

        segment.geometry.dispose();
        segment.material.dispose();
    }

    trail.fadeSegments = [];
}


function createFadeTrailSegment(
    trail,
    firstFrame,
    lastFrame,
    fadeFactor
) {
    /*
      A single short 1-Myr trajectory segment.

      `fadeFactor` is in [0, 1] and represents the age-dependent
      multiplier before applying the user-selected nominal trail opacity.
    */
    const positions = buildTrailPositions(
        trail.clusterIndex,
        firstFrame,
        lastFrame,
        -1
    );

    const geometry = new LineGeometry();
    geometry.setPositions(positions);

    const material = new LineMaterial({
        color: trail.material.color.clone(),

        linewidth: THREE.MathUtils.clamp(
            finiteNumber(params.trailLineWidth, 2.0),
            0.5,
            15.0
        ),

        transparent: true,

        opacity:
            THREE.MathUtils.clamp(
                finiteNumber(params.trailOpacity, 0.75),
                0.0,
                1.0
            )
            * fadeFactor,

        depthTest: true,
        depthWrite: false,

        worldUnits: false,
        toneMapped: false,
    });

    material.resolution.set(
        window.innerWidth,
        window.innerHeight
    );

    const line = new Line2(
        geometry,
        material
    );

    line.name = `${trail.line.name} age-fade segment`;

    /*
      Draw fade segments after the main path but before the cluster sphere.
    */
    line.renderOrder = 8;

    line.frustumCulled = false;

    /*
      Its final visibility is also controlled later in syncUniforms().
    */
    line.visible =
        Boolean(params.showTrails)
        && material.opacity > 0.001;

    scene.add(line);

    trail.fadeSegments.push({
        geometry,
        material,
        line,

        /*
          Store this so a GUI change to trail opacity can update the
          segment correctly without rebuilding the whole time path.
        */
        fadeFactor,
    });
}


function updateClusterTrails() {
    if (!clusterLayer) {
        return;
    }

    const currentFrame = clusterLayer.frameIndex;
    const presentFrame = clusterLayer.zeroTimeIndex;

    const currentTimeMyr = clusterLayer.timesMyr[
        currentFrame
    ];

    /*
      At t = 0, remove every dynamic fade segment and hide all trails.

      This explicitly fixes the small leftover segments you observed after
      travelling to the past and returning to the present.
    */
    if (currentFrame === presentFrame) {
        for (const trail of clusterLayer.trails) {
            trail.line.visible = false;

            /*
              Old single fade line is no longer used.
            */
            trail.fadeLine.visible = false;

            /*
              Remove all dynamically created one-Myr fade segments.
            */
            disposeFadeSegments(trail);
        }

        return;
    }

    const movingIntoFuture =
        currentFrame > presentFrame;

    for (const trail of clusterLayer.trails) {
        const cluster = clusterLayer.clusters[
            trail.clusterIndex
        ];

        /*
          Always clear the old set of individual fading segments before
          generating the correct new set for the current time.
        */
        disposeFadeSegments(trail);

        /*
          The old single fade line is permanently disabled.
        */
        trail.fadeLine.visible = false;

        /*
          ----------------------------------------------------------------
          Future trajectories
          ----------------------------------------------------------------

          Cluster ages do not truncate future integrations.
        */
        if (movingIntoFuture) {
            const positions = buildTrailPositions(
                trail.clusterIndex,
                presentFrame,
                currentFrame,
                1
            );

            replaceTrailGeometry(
                trail,
                'geometry',
                'line',
                positions
            );

            continue;
        }

        /*
          ----------------------------------------------------------------
          Past trajectories
          ----------------------------------------------------------------
        */
        const ageBoundaryTime = -cluster.ageMyr;

        const fadeEndTime =
            -cluster.ageMyr
            - CLUSTER_BIRTH_FADE_MYR;

        const ageBoundaryFrame =
            nearestClusterFrameIndex(
                ageBoundaryTime
            );

        const fadeEndFrame =
            nearestClusterFrameIndex(
                fadeEndTime
            );

        /*
          The selected time is younger than the nominal cluster age.

          Draw one ordinary path from t = 0 to the selected time.
        */
        if (currentTimeMyr >= ageBoundaryTime) {
            const positions = buildTrailPositions(
                trail.clusterIndex,
                presentFrame,
                currentFrame,
                -1
            );

            replaceTrailGeometry(
                trail,
                'geometry',
                'line',
                positions
            );

            continue;
        }

        /*
          The selected time is older than the cluster age.

          Draw the ordinary full-opacity path only down to the age limit.
        */
        const mainPositions = buildTrailPositions(
            trail.clusterIndex,
            presentFrame,
            ageBoundaryFrame,
            -1
        );

        replaceTrailGeometry(
            trail,
            'geometry',
            'line',
            mainPositions
        );

        /*
          Do not allow trail geometry beyond age + 5 Myr into the past.

          Because indices increase toward the future:
              currentFrame = selected past time
              fadeEndFrame = oldest permitted trail time

          max() selects the less-negative / allowed endpoint.
        */
        const cappedEndFrame = Math.max(
            currentFrame,
            fadeEndFrame
        );

        /*
          Build individual 1-Myr segments from the age boundary toward
          the selected ancient epoch.

          Example for age = 20 Myr and selected t = -23 Myr:

              main trail:
                  0 -> -20

              fading segments:
                  -20 -> -21
                  -21 -> -22
                  -22 -> -23
        */
        for (
            let firstFrame = ageBoundaryFrame;
            firstFrame > cappedEndFrame;
            firstFrame--
        ) {
            const lastFrame = firstFrame - 1;

            /*
              Opacity is based on the age of the older endpoint of this
              specific segment.

              Thus the sequence progressively fades away in time:
                  -20 -> -21 : relatively bright
                  -21 -> -22 : dimmer
                  ...
                  -24 -> -25 : nearly invisible
            */
            const segmentEndTime = clusterLayer.timesMyr[
                lastFrame
            ];

            const fadeFactor = clusterAgeOpacity(
                segmentEndTime,
                cluster.ageMyr
            );

            /*
              Do not create visually invisible geometry.
            */
            if (fadeFactor <= 0.001) {
                continue;
            }

            createFadeTrailSegment(
                trail,
                firstFrame,
                lastFrame,
                fadeFactor
            );
        }
    }
}


function updateClusterTrailStyle(force = false) {
    if (!clusterLayer) {
        return;
    }

    const lineWidth = THREE.MathUtils.clamp(
        finiteNumber(params.trailLineWidth, 2.0),
        0.5,
        15.0
    );

    const trailOpacity = THREE.MathUtils.clamp(
        finiteNumber(params.trailOpacity, 0.75),
        0.0,
        1.0
    );

    const colorByGroup = Boolean(
        params.colorClustersByGroup
    );

    const styleSignature = [
        lineWidth,
        trailOpacity,
        colorByGroup,
    ].join('|');

    if (
        !force
        && clusterLayer.trailStyleSignature
        === styleSignature
    ) {
        return;
    }

    for (const trail of clusterLayer.trails) {
        const cluster = clusterLayer.clusters[
            trail.clusterIndex
        ];

        const group = clusterLayer.groups[
            cluster.groupIndex
        ];

        const displayColour = new THREE.Color(
            colorByGroup
                ? getClusterGroupColour(group)
                : 0xffffff
        );

        /*
          Main trail appearance.
        */
        trail.material.color.copy(
            displayColour
        );

        trail.material.linewidth = lineWidth;
        trail.material.opacity = trailOpacity;


        /*
          Update every separately rendered fading segment.

          Each one has its own age-dependent fadeFactor, so all segments preserve
          the progressive opacity gradient when the user changes the global
          trail opacity, width, or group-colour setting.
        */
        for (const segment of trail.fadeSegments) {
            segment.material.color.copy(
                displayColour
            );

            segment.material.linewidth = lineWidth;

            segment.material.opacity =
                trailOpacity
                * segment.fadeFactor;
        }

        /*
          The old one-piece fade line is no longer used.
        */
        trail.fadeLine.visible = false;
    }

    clusterLayer.trailStyleSignature =
        styleSignature;
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
      Update cluster positions at the selected trajectory epoch.
    */
    updateClusterTransforms(true);

    /*
      Fade individual cluster spheres if the selected time is farther into
      the past than their age.
    */
    updateClusterAgeOpacity(canonicalTime);

    /*
      Rebuild paths from t = 0 to the selected time.
    */
    updateClusterTrails();

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
      LineMaterial widths are screen-space pixel widths, so each trajectory
      material must know the current browser dimensions.
    */
    if (clusterLayer) {
        for (const trail of clusterLayer.trails) {
            trail.material.resolution.set(
                window.innerWidth,
                window.innerHeight
            );

            /*
              Old fade material is retained only for compatibility, but hidden.
            */
            trail.fadeMaterial.resolution.set(
                window.innerWidth,
                window.innerHeight
            );

            /*
              Every dynamically-created fading segment needs the current canvas
              resolution because LineMaterial uses screen-space pixel width.
            */
            for (const segment of trail.fadeSegments) {
                segment.material.resolution.set(
                    window.innerWidth,
                    window.innerHeight
                );
            }
        }
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
