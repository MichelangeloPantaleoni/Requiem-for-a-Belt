import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19.2/dist/lil-gui.esm.min.js';


const MAX_STEPS = 512;

/*
  External timeline limits.

  The underlying trajectory files may contain a broader interval, but the
  interactive viewer exposes only this selected time range.
*/
const TIME_SLIDER_MIN_MYR = -50.0;
const TIME_SLIDER_MAX_MYR = 50.0;

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
  All coordinate/reference objects are rendered after cluster spheres.

  Their materials still use depth testing, so this does not force them
  visually in front. It merely ensures they test against the depth
  written by the clusters.
*/
const REFERENCE_RENDER_ORDER = 20;

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

/*
  Galactic z-axis visibility depends on the camera elevation relative
  to the Galactic x-y plane.

  At low elevation, the z-axis is useful and fully visible.

  At high elevation, it is nearly aligned with the viewing direction
  and becomes visually distracting, so it fades away.
*/
const Z_AXIS_FULL_OPACITY_ANGLE_DEG = 15.0;
const Z_AXIS_FADE_END_DEG = 50.0;

/*
  Solar-circle guide.

  The local LSR frame moves around the Galaxy at an assumed circular
  speed of 236 km/s and Galactocentric radius of 8.122 kpc.

  This is a geometric orientation aid, not an independently integrated
  orbit model.
*/
const SOLAR_CIRCLE_RADIUS_PC = 8122.0;

const SOLAR_CIRCLE_SPEED_KM_S = 236.0;

/*
  1 km/s = approximately 1.022712165 pc/Myr.
*/
const KM_S_TO_PC_MYR = 1.022712165;

/*
  Appearance.
*/
const SOLAR_CIRCLE_OPACITY = 0.75;

const SOLAR_CIRCLE_LINE_WIDTH_PX = 2.0;

const SOLAR_CIRCLE_TICK_INTERVAL_DEG = 10.0;

const SOLAR_CIRCLE_TICK_LENGTH_PC = 85.0;

const GALACTIC_CENTRE_GLOW_DIAMETER_PC = 1000.0;

/*
  Solar Galactocentric-radius reference line and label controls.
*/
const GALACTIC_RADIUS_LINE_NOMINAL_OPACITY = 0.75;

const GALACTIC_RADIUS_LABEL_NOMINAL_OPACITY = 1.0;

/*
  Text raster resolution and physical displayed size.

  Increasing FONT_SIZE_PX improves texture sharpness.
  Increasing LABEL_HEIGHT_PC makes the label physically larger.
*/
const GALACTIC_RADIUS_LABEL_FONT_SIZE_PX = 130;
const GALACTIC_RADIUS_LABEL_HEIGHT_PC = 140.0;

/*
  Separation between the radius line and the nearest edge of the
  "flag" label, in physical pc.
*/
const GALACTIC_RADIUS_LABEL_LINE_SEPARATION_PC = 70.0;

/*
  Distance-based attenuation relative to the centre of the LSR square.

  At or below 2 kpc: fully hidden.
  At or above 3 kpc: fully visible.
*/
const GALACTIC_RADIUS_REFERENCE_FADE_START_PC = 2700.0;
const GALACTIC_RADIUS_REFERENCE_FADE_END_PC = 4000.0;

/*
  Shift the rendered 3-D scene slightly upward in the browser viewport.

  This is a projection/framing adjustment only. It does not alter:
  - camera.position;
  - controls.target;
  - the point about which OrbitControls rotates.

  Increase this value if the bottom x_LSR label still sits too close to
  the time-slider readout.
*/
const CAMERA_VERTICAL_VIEW_OFFSET_PX = 45;

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
/* SUN GLOW SHADERS                                                           */
/* -------------------------------------------------------------------------- */

/*
  A perspective-scaled glowing point.

  Unlike the older cluster point sprites, this point has an apparent
  size determined by its physical diameter and camera distance:

      apparent size ∝ physical diameter / camera distance

  Therefore it naturally grows larger as the camera gets closer.
*/
const SUN_VERTEX_SHADER = /* glsl */ `
precision highp float;

uniform float uDiameterPc;
uniform float uProjectionScale;
uniform float uMaxPointSize;

void main() {
    vec4 mvPosition = modelViewMatrix * vec4(
        position,
        1.0
    );

    /*
      Camera-space depth.

      In Three.js camera coordinates, objects in front of the camera
      generally have negative z values.
    */
    float depth = max(-mvPosition.z, 0.001);

    /*
      Convert physical Sun diameter in pc into screen-space pixels.

      uProjectionScale is based on camera FOV and drawing-buffer height.
    */
    float pointSize =
        uDiameterPc
        * uProjectionScale
        / depth;

    gl_PointSize = clamp(
        pointSize,
        2.0,
        uMaxPointSize
    );

    gl_Position = projectionMatrix * mvPosition;
}
`;


const SUN_FRAGMENT_SHADER = /* glsl */ `
precision highp float;

void main() {
    /*
      Convert point coordinates into a circular local coordinate system.
    */
    vec2 localPoint =
        gl_PointCoord * 2.0 - 1.0;

    float radiusSquared = dot(
        localPoint,
        localPoint
    );

    if (radiusSquared > 1.0) {
        discard;
    }

    /*
      Compact bright core plus a soft yellow halo.
    */
    float core = exp(
        -18.0 * radiusSquared
    );

    float halo = exp(
        -3.4 * radiusSquared
    );

    /*
      Smoothly remove the square point-sprite boundary.
    */
    float edge =
        1.0 - smoothstep(
            0.78,
            1.0,
            sqrt(radiusSquared)
        );

    float alpha =
        (
            0.92 * core
            + 0.20 * halo
        )
        * edge;

    vec3 sunYellow = vec3(
        1.00,
        0.76,
        0.12
    );

    vec3 colour =
        sunYellow
        * (
            0.40 * halo
            + 1.35 * core
        );

    gl_FragColor = vec4(
        colour,
        alpha
    );
}
`;

/* -------------------------------------------------------------------------- */
/* GALACTIC-CENTRE GLOW SHADERS                                               */
/* -------------------------------------------------------------------------- */

const GALACTIC_CENTRE_VERTEX_SHADER = /* glsl */ `
precision highp float;

uniform float uDiameterPc;
uniform float uProjectionScale;
uniform float uMaxPointSize;

void main() {
    vec4 mvPosition = modelViewMatrix * vec4(
        position,
        1.0
    );

    float depth = max(
        -mvPosition.z,
        0.001
    );

    float pointSize =
        uDiameterPc
        * uProjectionScale
        / depth;

    gl_PointSize = clamp(
        pointSize,
        3.0,
        uMaxPointSize
    );

    gl_Position = projectionMatrix
        * mvPosition;
}
`;


const GALACTIC_CENTRE_FRAGMENT_SHADER = /* glsl */ `
precision highp float;

void main() {
    vec2 localPoint =
        gl_PointCoord * 2.0 - 1.0;

    float radiusSquared = dot(
        localPoint,
        localPoint
    );

    if (radiusSquared > 1.0) {
        discard;
    }

    /*
      Bright compact red core.
    */
    float core = exp(
        -50.0 * radiusSquared
    );

    /*
      Broader diffuse red halo.
    */
    float halo = exp(
        -2.35 * radiusSquared
    );

    float edge =
        1.0 - smoothstep(
            0.80,
            1.0,
            sqrt(radiusSquared)
        );

    float alpha =
        (
            0.95 * core
            + 0.24 * halo
        )
        * edge;

        /*
          Crimson halo with an intense warm-white central core.
        */
        vec3 crimsonColour = vec3(
            0.86,
            0.03,
            0.18
        );

        vec3 coreColour = vec3(
            1.00,
            0.72,
            0.78
        );

        vec3 colour =
            crimsonColour
            * 0.72
            * halo
            + coreColour
            * 1.0
            * core;

    gl_FragColor = vec4(
        colour,
        alpha
    );
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

/*
  Keep Three.js render-list ordering deterministic.

  This is important because coordinate/reference layers render after
  clusters, while still using depth testing for real 3-D occlusion.
*/
renderer.sortObjects = true;

app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x02050a);

let camera;
let controls;

let volumeMesh;
let uniforms;

/*
  Galactic-plane coordinate grid at z = 0.
*/
let galacticPlaneGrid = null;

/*
  Time-dependent Solar-circle guide in the Galactic plane.
*/
let solarCircleLayer = null;

/*
  The 3-D Gould Belt ellipse and its material.
  Keeping the material as a separate variable allows the GUI
  to update opacity and line width immediately.
*/
let gouldBeltLine;
let gouldBeltMaterial;


/*
  The 3-D Radcliffe wave model and its material.
*/
let radcliffeWaveLine;
let radcliffeWaveMaterial;


/*
  Contains the trajectory data, point geometry, material, attributes,
  and current trajectory frame.
*/
let clusterLayer = null;

/*
  Dynamic glowing Sun marker.
*/
let sunLayer = null;

let timeSliderInitialised = false;

let extent;
let centre;
let ranges;
let initialCameraPosition;

let params;

let renderPending = false;

/*
  Stores the current smooth camera-reset animation, if any.
*/
let cameraResetAnimationFrame = null;

/*
  Stores the smooth return-to-present-time animation, if active.
*/
let timeReturnAnimationFrame = null;

window.addEventListener('resize', onResize);

window.addEventListener(
    'keydown',
    onKeyDown
);

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
        sunMetadata,
        radcliffeWaveCsv,
    ] = await Promise.all([
        loadJSON('./data/density.json'),
        loadArrayBuffer('./data/density.u8'),
        new THREE.TextureLoader().loadAsync('./data/freeze.png'),

        loadJSON('./data/cluster_trajectories.json'),
        loadArrayBuffer('./data/cluster_trajectories.f32'),

        loadJSON('./data/sun_trajectory.json'),
        loadText('./data/Radcliffe_Wave_Best_Fit.csv'),
    ]);


    /*
      Validate and decode the cluster metadata and binary position array.
    */
    const clusterData = parseClusterDataset(
        clusterMetadata,
        clusterRawBuffer
    );

    const sunData = parseSunDataset(
        sunMetadata,
        clusterData.timesMyr
    );

    const radcliffeWavePositions = parseRadcliffeWaveCsv(
        radcliffeWaveCsv
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
    upper: 0.8,
    softness: 0.7,

    opacity: finiteNumber(defaults.opacity, 1.00),
    opticalDensity: finiteNumber(defaults.opticalDensity, 50.0),
    gamma: 1.5,

    steps: Math.round(finiteNumber(defaults.steps, 100)),

    /*
      ----------------------------------------------------------------
      Galactic-plane coordinate grid controls
      ----------------------------------------------------------------
    */
    showGalacticPlaneGrid: true,

    /*
      Opacity of the outer frame, major/minor ticks, tick labels,
      and axis labels.
    */
    gridFrameOpacity: 1.00,

    /*
      Opacity of the internal x/y grid lines.
    */
    gridLineOpacity: 0.30,

    /*
      Solar-circle Galactic-orientation guide.
    */
    showSolarCircle: true,

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
      Radcliffe wave model controls
      ----------------------------------------------------------------
    */
    showRadcliffeWave: false,

    radcliffeWaveLineWidth: 5.0,

    radcliffeWaveOpacity: 0.30,


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

    /*
      One common scale factor for all cluster markers.

      A value of 1.0 exactly preserves the current default min/max marker
      diameters and therefore preserves all relative size proportions.
    */
    clusterMarkerScale: 1.0,

    /*
      ----------------------------------------------------------------
      Sun controls
      ----------------------------------------------------------------
    */
    showSun: true,

    /*
      Physical apparent-diameter calibration in pc.

      This is not intended as the literal physical solar diameter. It is
      a visual marker diameter used to keep the Sun visible in a pc-scale
      Galactic visualization.
    */
    sunDiameterPc: 100.0,

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
      Move the visual framing slightly upward without changing the camera's
      physical position or orbit target.
    */
    updateCameraViewOffset();

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
      Restore the original camera position and orbit target.
    */
    params.resetView = () => {
        /*
          Prevent multiple overlapping reset animations.
        */
        if (cameraResetAnimationFrame !== null) {
            cancelAnimationFrame(
                cameraResetAnimationFrame
            );

            cameraResetAnimationFrame = null;
        }

        /*
          Two seconds, as requested.
        */
        const durationMs = 2000.0;

        const startingPosition =
            camera.position.clone();

        const startingTarget =
            controls.target.clone();

        const endingPosition =
            initialCameraPosition.clone();

        const endingTarget =
            centre.clone();

        const startTime = performance.now();

        /*
          Temporarily disable manual camera interaction while the reset
          animation is running.
        */
        controls.enabled = false;

        function smoothStep(value) {
            /*
              Smooth cubic easing:

                  0 -> 0
                  1 -> 1

              with zero velocity at both ends.
            */
            return value * value * (
                3.0 - 2.0 * value
            );
        }

        function animateReset(currentTime) {
            const rawProgress = THREE.MathUtils.clamp(
                (currentTime - startTime) / durationMs,
                0.0,
                1.0
            );

            const easedProgress = smoothStep(
                rawProgress
            );

            camera.position.lerpVectors(
                startingPosition,
                endingPosition,
                easedProgress
            );

            controls.target.lerpVectors(
                startingTarget,
                endingTarget,
                easedProgress
            );

            controls.update();

            requestRender();

            if (rawProgress < 1.0) {
                cameraResetAnimationFrame =
                    requestAnimationFrame(
                        animateReset
                    );
            } else {
                /*
                  Ensure exact final values, avoiding tiny accumulated
                  interpolation differences.
                */
                camera.position.copy(
                    endingPosition
                );

                controls.target.copy(
                    endingTarget
                );

                controls.enabled = true;

                controls.update();

                cameraResetAnimationFrame = null;

                requestRender();
            }
        }

        cameraResetAnimationFrame =
            requestAnimationFrame(
                animateReset
            );
    };

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
      Galactic-plane coordinate grid.

      It is added directly to the scene and remains independent from the
      density field, cluster trajectories, and time selection.
    */
    galacticPlaneGrid = createGalacticPlaneGrid();

    /*
      Solar-circle guide. It is updated every time the trajectory slider
      selects a new Myr epoch.
    */
    solarCircleLayer = createSolarCircleLayer();

    /*
      Make the entire LSR-coordinate system and the Galactic-reference
      system render after clusters while retaining normal depth testing.
    */
    setReferenceGroupRenderOrder(
        galacticPlaneGrid.root
    );

    setReferenceGroupRenderOrder(
        solarCircleLayer.root
    );

    /*
      Create the 3-D Gould Belt ellipse.

      It is added directly to `scene`, rather than to `volumeMesh`,
      because its coordinates are already physical Galactic Cartesian
      coordinates in pc.
    */
    gouldBeltLine = createGouldBeltModel();

    gouldBeltLine.renderOrder =
        REFERENCE_RENDER_ORDER;

    scene.add(gouldBeltLine);

    /*
      Create the static Radcliffe-wave best-fit model.

      Its visibility is controlled independently in the GUI, but its
      temporal fade matches the Gould Belt model.
    */
    radcliffeWaveLine = createRadcliffeWaveModel(
        radcliffeWavePositions
    );

    radcliffeWaveLine.renderOrder =
        REFERENCE_RENDER_ORDER;

    scene.add(radcliffeWaveLine);

    /*
      Create the dynamic point layer containing all clusters.
      The helper function adds the points to the Three.js scene.
    */
    clusterLayer = createClusterLayer(clusterData);

    sunLayer = createSunLayer(sunData);

    /*
      Start at t = 0 Myr.
    */
    setClusterFrameFromTime(
        params.clusterTime
    );

    /*
      Place the Sun at its t = 0 position initially.
    */
    updateSunFrame(
        clusterLayer.zeroTimeIndex
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
            .add(params, 'softness', 0.001, 1.0, 0.001)
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

    watched(
        clusterFolder
            .add(
                params,
                'clusterMarkerScale',
                0.25,
                5.0,
                0.05
            )
            .name('Marker size')
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
    /* 4. RADCLIFFE WAVE MODEL                                                 */
    /* ====================================================================== */

    const radcliffeFolder = gui.addFolder(
        'Radcliffe wave model [Konietzka et al. 2024]'
    );

    watched(
        radcliffeFolder
            .add(params, 'showRadcliffeWave')
            .name('Show Radcliffe wave model')
    );

    watched(
        radcliffeFolder
            .add(
                params,
                'radcliffeWaveLineWidth',
                1.0,
                15.0,
                0.25
            )
            .name('Line width [px]')
    );

    watched(
        radcliffeFolder
            .add(
                params,
                'radcliffeWaveOpacity',
                0.0,
                1.0,
                0.01
            )
            .name('Opacity')
    );


    /* ====================================================================== */
    /* 5. COORDINATES AND OTHEWR REFERENCES                                               */
    /* ====================================================================== */

    const gridFolder = gui.addFolder(
        'Coordinates and other references'
    );

    watched(
        gridFolder
            .add(params, 'showGalacticPlaneGrid')
            .name('Show LSR coordinates')
    );

    watched(
        gridFolder
            .add(
                params,
                'gridFrameOpacity',
                0.0,
                1.0,
                0.01
            )
            .name('Frame, ticks and labels')
    );

    watched(
        gridFolder
            .add(
                params,
                'gridLineOpacity',
                0.0,
                1.0,
                0.01
            )
            .name('Internal grid lines')
    );

    watched(
        gridFolder
            .add(params, 'showSolarCircle')
            .name('Show Galactic references')
    );

    watched(
        gridFolder
            .add(params, 'showSun')
            .name('Show Sun')
    );

    watched(
        gridFolder
            .add(
                params,
                'sunDiameterPc',
                1.0,
                300.0,
                1.0
            )
            .name('Sun glow diameter [pc]')
    );

    gridFolder
        .add(params, 'resetView')
        .name('Reset camera view [R]');


    /*
      Keep every top-level scientific menu collapsed at startup.
    */
    densityFolder.close();
    clusterFolder.close();
    gouldFolder.close();
    radcliffeFolder.close();
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
      Move the Sun to the matching trajectory epoch.
    */
    updateSunFrame(clusterFrameIndex);

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
      Rotate the local Solar-circle guide to the selected LSR epoch.
    */
    updateSolarCircle(selectedTimeMyr);

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
    /*
      The Radcliffe wave follows the same temporal visibility law as the
      Gould Belt: fully visible at t = 0 and faded out by |t| = 3 Myr.
    */
    const radcliffeWaveIsVisible =
        Boolean(params.showRadcliffeWave)
        && gouldBeltFade > 0.001;

    const lower = THREE.MathUtils.clamp(
        finiteNumber(params.lower, 0.5),
        0.0,
        0.995
    );

    const upper = THREE.MathUtils.clamp(
        finiteNumber(params.upper, 0.8),
        lower + 0.005,
        1.0
    );

    uniforms.uLower.value = lower;
    uniforms.uUpper.value = upper;

    uniforms.uSoftness.value = THREE.MathUtils.clamp(
        finiteNumber(params.softness, 1.0),
        0.001,
        1.0
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
        finiteNumber(params.gamma, 1.5),
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
      The Sun remains independently visible at every selected epoch.
    */
    sunLayer.points.visible = Boolean(
        params.showSun
    );

    sunLayer.material.uniforms.uDiameterPc.value =
        THREE.MathUtils.clamp(
            finiteNumber(params.sunDiameterPc, 100.0),
            1.0,
            300.0
        );

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
      Gould Belt model.
    */
    gouldBeltLine.visible =
        gouldBeltIsVisible;

    gouldBeltMaterial.linewidth =
        THREE.MathUtils.clamp(
            finiteNumber(
                params.gouldBeltLineWidth,
                5.5
            ),
            0.5,
            30.0
        );

    const nominalGouldBeltOpacity =
        THREE.MathUtils.clamp(
            finiteNumber(
                params.gouldBeltOpacity,
                0.50
            ),
            0.0,
            1.0
        );

    gouldBeltMaterial.opacity =
        nominalGouldBeltOpacity
        * gouldBeltFade;


    /*
      Radcliffe-wave model.

      It has independent GUI controls, but shares the same temporal fading
      factor as Gould's Belt.
    */
    radcliffeWaveLine.visible =
        radcliffeWaveIsVisible;

    radcliffeWaveMaterial.linewidth =
        THREE.MathUtils.clamp(
            finiteNumber(
                params.radcliffeWaveLineWidth,
                5.0
            ),
            0.5,
            30.0
        );

    const nominalRadcliffeWaveOpacity =
        THREE.MathUtils.clamp(
            finiteNumber(
                params.radcliffeWaveOpacity,
                0.30
            ),
            0.0,
            1.0
        );

    radcliffeWaveMaterial.opacity =
        nominalRadcliffeWaveOpacity
        * gouldBeltFade;

    /*
      Galactic plane grid is independent from time and all scientific layers.
    */
    if (galacticPlaneGrid) {
        galacticPlaneGrid.root.visible = Boolean(
            params.showGalacticPlaneGrid
        );

        updateGalacticPlaneGridStyle();

        /*
          Labels move to the border sides nearest the camera as the user
          rotates around the scene.
        */
        updateGalacticPlaneGridLabels();
    }

    /*
      Solar circle is independent from the square coordinate grid.
    */
    updateSolarCircleStyle();

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


function createRadcliffeWaveModel(
    positions
) {
    /*
      `positions` contains ordered x/y/z coordinates from the CSV:

          x0, y0, z0,
          x1, y1, z1,
          ...

      Line2 joins them in their CSV order, producing the oscillating
      Radcliffe-wave curve.
    */
    const geometry = new LineGeometry();

    geometry.setPositions(
        positions
    );

    /*
      Crimson, matching the gamma Vel family colour and the request for
      a visually distinct Radcliffe-wave curve.
    */
    radcliffeWaveMaterial = new LineMaterial({
        color: 0xdc143c,

        linewidth: params.radcliffeWaveLineWidth,

        transparent: true,
        opacity: params.radcliffeWaveOpacity,

        depthTest: true,
        depthWrite: false,

        worldUnits: false,

        toneMapped: false,
    });

    radcliffeWaveMaterial.resolution.set(
        window.innerWidth,
        window.innerHeight
    );

    const line = new Line2(
        geometry,
        radcliffeWaveMaterial
    );

    line.name = 'Radcliffe wave model';

    /*
      The wave may extend outside the local density volume, so avoid
      accidental disappearance from stale automatic bounds.
    */
    line.frustumCulled = false;

    return line;
}

function setReferenceGroupRenderOrder(root) {
    if (!root) {
        return;
    }

    /*
      Group renderOrder is inherited by descendants during Three.js
      render-list construction.

      Nested groups are included because the z-axis has nested tick and
      label groups.
    */
    root.traverse((object) => {
        if (object.isGroup) {
            object.renderOrder =
                REFERENCE_RENDER_ORDER;
        }
    });
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

    const datasetFirstTime = clusterLayer.timesMyr[0];

    const datasetLastTime = clusterLayer.timesMyr[
        clusterLayer.timesMyr.length - 1
    ];

    const firstTime = THREE.MathUtils.clamp(
        TIME_SLIDER_MIN_MYR,
        datasetFirstTime,
        datasetLastTime
    );

    const lastTime = THREE.MathUtils.clamp(
        TIME_SLIDER_MAX_MYR,
        datasetFirstTime,
        datasetLastTime
    );

    const currentTime = THREE.MathUtils.clamp(
        clusterLayer.timesMyr[
            nearestClusterFrameIndex(
                params.clusterTime
            )
        ],
        firstTime,
        lastTime
    );

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

    const datasetFirstTime = clusterLayer.timesMyr[0];

    const datasetLastTime = clusterLayer.timesMyr[
        clusterLayer.timesMyr.length - 1
    ];

    const firstTime = THREE.MathUtils.clamp(
        TIME_SLIDER_MIN_MYR,
        datasetFirstTime,
        datasetLastTime
    );

    const lastTime = THREE.MathUtils.clamp(
        TIME_SLIDER_MAX_MYR,
        datasetFirstTime,
        datasetLastTime
    );

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

      For the -50 ... +50 Myr visible timeline this creates marks at:
      -50, -40, ..., -10, 0, +10, ..., +50 Myr.
    */
    for (
        let time = TIME_SLIDER_MIN_MYR;
        time <= TIME_SLIDER_MAX_MYR;
        time += 10
    ) {
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

    /*
      Restrict the user-facing timeline to -50 ... +50 Myr, while ensuring
      the requested limits remain valid for the loaded trajectory data.
    */
    const datasetFirstTime = clusterLayer.timesMyr[0];

    const datasetLastTime = clusterLayer.timesMyr[
        clusterLayer.timesMyr.length - 1
    ];

    const firstTime = THREE.MathUtils.clamp(
        TIME_SLIDER_MIN_MYR,
        datasetFirstTime,
        datasetLastTime
    );

    const lastTime = THREE.MathUtils.clamp(
        TIME_SLIDER_MAX_MYR,
        datasetFirstTime,
        datasetLastTime
    );

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

    /*
      Avoid leaving a visible keyboard-focus ring around the timeline after
      a mouse/touch drag. Keyboard shortcuts still work even without this,
      but it makes the interaction feel cleaner.
    */
    timeSlider.addEventListener(
        'pointerup',
        () => {
            if (document.activeElement === timeSlider) {
                timeSlider.blur();
            }
        }
    );

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


/* -------------------------------------------------------------------------- */
/* RADCLIFFE WAVE DATA                                                        */
/* -------------------------------------------------------------------------- */

function parseRadcliffeWaveCsv(csvText) {
    if (typeof csvText !== 'string') {
        throw new Error(
            'Radcliffe-wave CSV could not be read as text.'
        );
    }

    /*
      Remove a possible UTF-8 byte-order mark, split into rows, and
      discard empty lines.
    */
    const rows = csvText
        .replace(/^\uFEFF/, '')
        .split(/\r?\n/)
        .map((row) => row.trim())
        .filter((row) => row.length > 0);

    if (rows.length < 3) {
        throw new Error(
            'Radcliffe-wave CSV must contain a header and at least two data rows.'
        );
    }

    const cleanCell = (value) => {
        return String(value)
            .trim()
            .replace(/^"|"$/g, '');
    };

    /*
      Read header names case-insensitively.

      Expected CSV columns:
          x,y,z
    */
    const headers = rows[0]
        .split(',')
        .map((value) => {
            return cleanCell(value).toLowerCase();
        });

    const xIndex = headers.indexOf('x');
    const yIndex = headers.indexOf('y');
    const zIndex = headers.indexOf('z');

    if (
        xIndex < 0
        || yIndex < 0
        || zIndex < 0
    ) {
        throw new Error(
            'Radcliffe-wave CSV must contain x, y, and z columns.'
        );
    }

    const positionValues = [];

    for (
        let rowIndex = 1;
        rowIndex < rows.length;
        rowIndex++
    ) {
        const columns = rows[rowIndex].split(',');

        const x = Number(
            cleanCell(columns[xIndex] ?? '')
        );

        const y = Number(
            cleanCell(columns[yIndex] ?? '')
        );

        const z = Number(
            cleanCell(columns[zIndex] ?? '')
        );

        if (
            !Number.isFinite(x)
            || !Number.isFinite(y)
            || !Number.isFinite(z)
        ) {
            throw new Error(
                'Radcliffe-wave CSV contains an invalid coordinate '
                + `at row ${rowIndex + 1}.`
            );
        }

        positionValues.push(
            x,
            y,
            z
        );
    }

    if (positionValues.length < 6) {
        throw new Error(
            'Radcliffe-wave CSV does not contain enough valid points.'
        );
    }

    return new Float32Array(
        positionValues
    );
}


/* -------------------------------------------------------------------------- */
/* SUN TRAJECTORY DATA                                                        */
/* -------------------------------------------------------------------------- */

function parseSunDataset(
    metadata,
    referenceTimesMyr
) {
    if (
        !metadata
        || !Array.isArray(metadata.timesMyr)
        || !Array.isArray(metadata.positionsPc)
    ) {
        throw new Error(
            'sun_trajectory.json has an invalid structure.'
        );
    }

    const timesMyr = metadata.timesMyr.map(Number);

    if (
        timesMyr.length !== referenceTimesMyr.length
    ) {
        throw new Error(
            'Sun trajectory time grid does not match '
            + 'the cluster trajectory time grid.'
        );
    }

    for (
        let index = 0;
        index < referenceTimesMyr.length;
        index++
    ) {
        if (
            Math.abs(
                timesMyr[index]
                - referenceTimesMyr[index]
            ) > 1.0e-8
        ) {
            throw new Error(
                'Sun trajectory times do not match '
                + 'the cluster trajectory times.'
            );
        }
    }

    if (
        metadata.positionsPc.length
        !== timesMyr.length
    ) {
        throw new Error(
            'sun_trajectory.json has a different number of '
            + 'positions and time values.'
        );
    }

    const positions = new Float32Array(
        timesMyr.length * 3
    );

    for (
        let index = 0;
        index < metadata.positionsPc.length;
        index++
    ) {
        const position = metadata.positionsPc[index];

        if (
            !Array.isArray(position)
            || position.length !== 3
        ) {
            throw new Error(
                `Invalid Sun position at time index ${index}.`
            );
        }

        const x = Number(position[0]);
        const y = Number(position[1]);
        const z = Number(position[2]);

        if (
            !Number.isFinite(x)
            || !Number.isFinite(y)
            || !Number.isFinite(z)
        ) {
            throw new Error(
                `Non-finite Sun coordinate at time index ${index}.`
            );
        }

        positions[3 * index + 0] = x;
        positions[3 * index + 1] = y;
        positions[3 * index + 2] = z;
    }

    return {
        timesMyr,
        positions,
    };
}


/* -------------------------------------------------------------------------- */
/* SUN GLOW MARKER                                                            */
/* -------------------------------------------------------------------------- */

function createSunLayer(dataset) {
    const geometry = new THREE.BufferGeometry();

    const positionAttribute =
        new THREE.BufferAttribute(
            new Float32Array([0, 0, 0]),
            3
        );

    positionAttribute.setUsage(
        THREE.DynamicDrawUsage
    );

    geometry.setAttribute(
        'position',
        positionAttribute
    );

    const gl = renderer.getContext();

    const pointSizeRange = gl.getParameter(
        gl.ALIASED_POINT_SIZE_RANGE
    );

    const material = new THREE.ShaderMaterial({
        uniforms: {
            uDiameterPc: {
                value: params.sunDiameterPc,
            },

            /*
              Updated after resize because it depends on drawing-buffer
              dimensions and camera FOV.
            */
            uProjectionScale: {
                value: 1.0,
            },

            uMaxPointSize: {
                value: Number(pointSizeRange[1]),
            },
        },

        vertexShader: SUN_VERTEX_SHADER,
        fragmentShader: SUN_FRAGMENT_SHADER,

        transparent: true,

        depthTest: true,
        depthWrite: false,

        /*
          Additive blending gives the yellow halo a luminous appearance.
        */
        blending: THREE.AdditiveBlending,

        toneMapped: false,
    });

    const points = new THREE.Points(
        geometry,
        material
    );

    points.name = 'Sun';

    /*
      The Sun may move beyond the initial camera bounds, so do not permit
      stale automatic bounds to hide it.
    */
    points.frustumCulled = false;

    /*
      Render above trails and cluster spheres.
    */
    points.renderOrder = 12;

    scene.add(points);

    const layer = {
        timesMyr: dataset.timesMyr,
        positions: dataset.positions,

        geometry,
        positionAttribute,
        material,
        points,

        frameIndex: -1,
    };

    updateSunProjectionScale(layer);

    return layer;
}


function updateSunProjectionScale(layer = sunLayer) {
    if (!layer || !camera) {
        return;
    }

    const drawingBufferSize =
        renderer.getDrawingBufferSize(
            new THREE.Vector2()
        );

    const verticalFovRadians =
        THREE.MathUtils.degToRad(
            camera.fov
        );

    /*
      Perspective projection scale in actual drawing-buffer pixels.
    */
    const projectionScale =
        drawingBufferSize.y
        / (
            2.0
            * Math.tan(
                verticalFovRadians * 0.5
            )
        );

    layer.material.uniforms.uProjectionScale.value =
        projectionScale;
}


function updateSunFrame(frameIndex) {
    if (!sunLayer) {
        return;
    }

    if (sunLayer.frameIndex === frameIndex) {
        return;
    }

    const sourceIndex = frameIndex * 3;

    sunLayer.positionAttribute.array[0] =
        sunLayer.positions[sourceIndex + 0];

    sunLayer.positionAttribute.array[1] =
        sunLayer.positions[sourceIndex + 1];

    sunLayer.positionAttribute.array[2] =
        sunLayer.positions[sourceIndex + 2];

    sunLayer.positionAttribute.needsUpdate = true;

    sunLayer.frameIndex = frameIndex;
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

    /*
      Preserve the current default marker-size prescription. The new GUI
      control only multiplies both values by one common scale factor.
    */
    const baseMinMarkerDiameterPx =
        THREE.MathUtils.clamp(
            finiteNumber(
                dataset.defaultControls.minMarkerSize,
                9.0
            ),
            1.0,
            80.0
        );

    const baseMaxMarkerDiameterPx =
        Math.max(
            baseMinMarkerDiameterPx,
            THREE.MathUtils.clamp(
                finiteNumber(
                    dataset.defaultControls.maxMarkerSize,
                    22.0
                ),
                1.0,
                120.0
            )
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

        baseMinMarkerDiameterPx,
        baseMaxMarkerDiameterPx,

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

    const markerScale =
        THREE.MathUtils.clamp(
            finiteNumber(
                params.clusterMarkerScale,
                1.0
            ),
            0.25,
            5.0
        );

    /*
      Scale both the minimum and maximum default marker diameters by the
      same factor, preserving the existing relative size prescription.
    */
    const minDiameterPx =
        clusterLayer.baseMinMarkerDiameterPx
        * markerScale;

    const maxDiameterPx =
        clusterLayer.baseMaxMarkerDiameterPx
        * markerScale;

    const colorByGroup = Boolean(
        params.colorClustersByGroup
    );

    const viewportHeight = Math.max(
        renderer.domElement.clientHeight,
        1
    );

    const styleSignature = [
        markerScale,
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

/* -------------------------------------------------------------------------- */
/* GALACTIC-PLANE GRID                                                        */
/* -------------------------------------------------------------------------- */

function makeLineSegmentsObject(
    positions,
    material,
    renderOrder = 1
) {
    const geometry = new THREE.BufferGeometry();

    geometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(
            positions,
            3
        )
    );

    const lines = new THREE.LineSegments(
        geometry,
        material
    );

    lines.renderOrder = renderOrder;

    return lines;
}


function makeGridTextSprite(
    text,
    options = {}
) {
    const {
        fontSize = 54,
        fontFamily = 'Georgia, Times New Roman, serif',
        colour = 'rgba(245, 250, 255, 1.0)',
        padding = 18,
        scaleY = 34,
    } = options;

    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    /*
      Use a high-resolution canvas so labels remain reasonably sharp.
    */
    const deviceScale = 4;

    context.font = `${fontSize}px ${fontFamily}`;

    const textWidth = Math.ceil(
        context.measureText(text).width
    );

    canvas.width = Math.ceil(
        (textWidth + 2 * padding) * deviceScale
    );

    canvas.height = Math.ceil(
        (fontSize + 2 * padding) * deviceScale
    );

    context.scale(
        deviceScale,
        deviceScale
    );

    context.font = `${fontSize}px ${fontFamily}`;
    context.textAlign = 'center';
    context.textBaseline = 'middle';

    /*
      Soft dark shadow keeps labels readable over bright density regions.
    */
    context.shadowColor = 'rgba(0, 0, 0, 0.90)';
    context.shadowBlur = 7;
    context.shadowOffsetX = 1;
    context.shadowOffsetY = 1;

    context.fillStyle = colour;

    context.fillText(
        text,
        canvas.width / (2 * deviceScale),
        canvas.height / (2 * deviceScale)
    );

    const texture = new THREE.CanvasTexture(canvas);

    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;

    const material = new THREE.SpriteMaterial({
        map: texture,

        transparent: true,
        opacity: 1.0,

        depthTest: true,
        depthWrite: false,

        toneMapped: false,
    });

    const sprite = new THREE.Sprite(material);


    /*
      Sprite dimensions are measured in scene pc units. Sprites naturally
      become larger on screen as the camera approaches them.
    */

    /*
      Preserve the canvas aspect ratio. This prevents characters such as
      "0" from appearing horizontally compressed.
    */
    const aspectRatio =
        canvas.width / canvas.height;

    sprite.scale.set(
        scaleY * aspectRatio,
        scaleY,
        1.0
    );

    sprite.renderOrder = 6;

    return sprite;
}


function formatGridCoordinate(value) {
    /*
      Coordinates are generated at exact 200-pc intervals, but round
      defensively to avoid labels such as "-199.999999".
    */
    const rounded = Math.round(value);

    return String(rounded);
}

function makeFlatTextLabel(
    text,
    options = {}
) {
    const {
        fontSize = 130,
        fontFamily = 'Georgia, Times New Roman, serif',
        colour = 'rgba(235, 245, 255, 1.0)',
        padding = 18,

        /*
          Physical height in pc. The width follows naturally from the
          rendered canvas aspect ratio.
        */
        heightPc = 105,
    } = options;

    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    const deviceScale = 4;

    context.font = `${fontSize}px ${fontFamily}`;

    const textWidth = Math.ceil(
        context.measureText(text).width
    );

    const logicalWidth =
        textWidth + 2 * padding;

    const logicalHeight =
        fontSize + 2 * padding;

    canvas.width = Math.ceil(
        logicalWidth * deviceScale
    );

    canvas.height = Math.ceil(
        logicalHeight * deviceScale
    );

    context.scale(
        deviceScale,
        deviceScale
    );

    context.font = `${fontSize}px ${fontFamily}`;
    context.textAlign = 'center';
    context.textBaseline = 'middle';

    context.shadowColor = 'rgba(0, 0, 0, 0.90)';
    context.shadowBlur = 7;
    context.shadowOffsetX = 1;
    context.shadowOffsetY = 1;

    context.fillStyle = colour;

    context.fillText(
        text,
        logicalWidth * 0.5,
        logicalHeight * 0.5
    );

    const texture = new THREE.CanvasTexture(
        canvas
    );

    texture.colorSpace = THREE.SRGBColorSpace;

    texture.anisotropy = Math.min(
        renderer.capabilities.getMaxAnisotropy(),
        8
    );

    texture.needsUpdate = true;

    const aspectRatio =
        canvas.width / canvas.height;

    const geometry = new THREE.PlaneGeometry(
        heightPc * aspectRatio,
        heightPc
    );

    const material = new THREE.MeshBasicMaterial({
        map: texture,

        transparent: true,
        opacity: 1.0,

        side: THREE.DoubleSide,

        depthTest: true,
        depthWrite: false,

        toneMapped: false,
    });

    const mesh = new THREE.Mesh(
        geometry,
        material
    );

    mesh.renderOrder = 6;

    return {
        mesh,
        geometry,
        material,
        texture,
    };
}

function createGalacticPlaneGrid() {
    const root = new THREE.Group();

    root.name = 'Galactic plane coordinate grid';

    /*
      The actual geometric grid sits exactly on the Galactic plane.
    */
    const zPlane = -20.0;

    const xMin = ranges.x[0];
    const xMax = ranges.x[1];

    const yMin = ranges.y[0];
    const yMax = ranges.y[1];

    const majorIntervalPc = 200.0;
    const minorIntervalPc = 50.0;

    /*
      Tick lengths point inward from the square border.
    */
    const majorTickLengthPc = 26.0;
    const minorTickLengthPc = 13.0;

    /*
      Labels are lifted slightly above z = 0 to keep them visually
      separate from the grid itself.
    */
    const labelZ = -14.0;

    /*
      ----------------------------------------------------------------
      Materials
      ----------------------------------------------------------------
    */

    const gridMaterial = new THREE.LineBasicMaterial({
        color: 0x83c7ff,

        transparent: true,
        opacity: params.gridLineOpacity,

        depthTest: true,
        depthWrite: false,

        toneMapped: false,
    });

    const tickMaterial = new THREE.LineBasicMaterial({
        color: 0xe8f5ff,

        transparent: true,
        opacity: params.gridFrameOpacity,

        depthTest: true,
        depthWrite: false,

        toneMapped: false,
    });

    /*
      Broad translucent blue line behind the main frame:
      this approximates the glow used on the external time slider.
    */
    const spineGlowMaterial = new LineMaterial({
        color: 0x8fd8ff,

        linewidth: 8.0,

        transparent: true,
        opacity: params.gridFrameOpacity * 0.20,

        depthTest: true,
        depthWrite: false,

        blending: THREE.AdditiveBlending,

        worldUnits: false,
        toneMapped: false,
    });

    const spineMaterial = new LineMaterial({
        color: 0xf2fbff,

        linewidth: 2.0,

        transparent: true,
        opacity: params.gridFrameOpacity,

        depthTest: true,
        depthWrite: false,

        worldUnits: false,
        toneMapped: false,
    });

    spineGlowMaterial.resolution.set(
        window.innerWidth,
        window.innerHeight
    );

    spineMaterial.resolution.set(
        window.innerWidth,
        window.innerHeight
    );


    /*
      ----------------------------------------------------------------
      Internal major grid lines: every 200 pc
      ----------------------------------------------------------------
    */

    const gridPositions = [];

    const firstMajorX =
        Math.ceil(xMin / majorIntervalPc)
        * majorIntervalPc;

    const firstMajorY =
        Math.ceil(yMin / majorIntervalPc)
        * majorIntervalPc;

    for (
        let x = firstMajorX;
        x <= xMax + 1.0e-6;
        x += majorIntervalPc
    ) {
        /*
          Boundary grid lines are already represented by the outer frame.
        */
        if (
            x <= xMin + 1.0e-6
            || x >= xMax - 1.0e-6
        ) {
            continue;
        }

        gridPositions.push(
            x, yMin, zPlane,
            x, yMax, zPlane
        );
    }

    for (
        let y = firstMajorY;
        y <= yMax + 1.0e-6;
        y += majorIntervalPc
    ) {
        if (
            y <= yMin + 1.0e-6
            || y >= yMax - 1.0e-6
        ) {
            continue;
        }

        gridPositions.push(
            xMin, y, zPlane,
            xMax, y, zPlane
        );
    }

    const gridLines = makeLineSegmentsObject(
        gridPositions,
        gridMaterial,
        1
    );

    root.add(gridLines);


    /*
      ----------------------------------------------------------------
      Outer glowing frame
      ----------------------------------------------------------------
    */

    const framePositions = new Float32Array([
        xMin, yMin, zPlane,
        xMax, yMin, zPlane,

        xMax, yMin, zPlane,
        xMax, yMax, zPlane,

        xMax, yMax, zPlane,
        xMin, yMax, zPlane,

        xMin, yMax, zPlane,
        xMin, yMin, zPlane,
    ]);

    const frameGlowGeometry = new LineGeometry();
    frameGlowGeometry.setPositions(
        framePositions
    );

    const frameGeometry = new LineGeometry();
    frameGeometry.setPositions(
        framePositions
    );

    const frameGlow = new Line2(
        frameGlowGeometry,
        spineGlowMaterial
    );

    const frame = new Line2(
        frameGeometry,
        spineMaterial
    );

    frameGlow.renderOrder = 2;
    frame.renderOrder = 3;

    frameGlow.frustumCulled = false;
    frame.frustumCulled = false;

    root.add(frameGlow);
    root.add(frame);


    /*
      ----------------------------------------------------------------
      Major/minor inward ticks
      ----------------------------------------------------------------
    */

    const majorTickPositions = [];
    const minorTickPositions = [];

    const majorXValues = [];
    const majorYValues = [];

    const firstMinorX =
        Math.ceil(xMin / minorIntervalPc)
        * minorIntervalPc;

    const firstMinorY =
        Math.ceil(yMin / minorIntervalPc)
        * minorIntervalPc;

    function isMajorCoordinate(value) {
        const scaled = value / majorIntervalPc;

        return Math.abs(
            scaled - Math.round(scaled)
        ) < 1.0e-6;
    }

    /*
      Ticks along the x direction:
      - bottom edge points upward/inward;
      - top edge points downward/inward.
    */
    for (
        let x = firstMinorX;
        x <= xMax + 1.0e-6;
        x += minorIntervalPc
    ) {
        if (
            x <= xMin + 1.0e-6
            || x >= xMax - 1.0e-6
        ) {
            continue;
        }

        const major = isMajorCoordinate(x);

        const length = major
            ? majorTickLengthPc
            : minorTickLengthPc;

        const target = major
            ? majorTickPositions
            : minorTickPositions;

        target.push(
            x, yMin, zPlane,
            x, yMin + length, zPlane,

            x, yMax, zPlane,
            x, yMax - length, zPlane
        );

        if (major) {
            majorXValues.push(x);
        }
    }

    /*
      Ticks along the y direction:
      - left edge points right/inward;
      - right edge points left/inward.
    */
    for (
        let y = firstMinorY;
        y <= yMax + 1.0e-6;
        y += minorIntervalPc
    ) {
        if (
            y <= yMin + 1.0e-6
            || y >= yMax - 1.0e-6
        ) {
            continue;
        }

        const major = isMajorCoordinate(y);

        const length = major
            ? majorTickLengthPc
            : minorTickLengthPc;

        const target = major
            ? majorTickPositions
            : minorTickPositions;

        target.push(
            xMin, y, zPlane,
            xMin + length, y, zPlane,

            xMax, y, zPlane,
            xMax - length, y, zPlane
        );

        if (major) {
            majorYValues.push(y);
        }
    }

    const minorTicks = makeLineSegmentsObject(
        minorTickPositions,
        tickMaterial,
        3
    );

    const majorTicks = makeLineSegmentsObject(
        majorTickPositions,
        tickMaterial,
        4
    );

    root.add(minorTicks);
    root.add(majorTicks);


    /*
      ----------------------------------------------------------------
      Numerical coordinate labels
      ----------------------------------------------------------------

      The label positions are updated continuously so they stay on the
      x/y sides closest to the current camera position.
    */

    const xTickLabels = majorXValues.map((xValue) => {
        const sprite = makeGridTextSprite(
            formatGridCoordinate(xValue),
            {
                fontSize: 90,
                scaleY: 50,
            }
        );

        root.add(sprite);

        return {
            value: xValue,
            sprite,
        };
    });

    const yTickLabels = majorYValues.map((yValue) => {
        const sprite = makeGridTextSprite(
            formatGridCoordinate(yValue),
            {
                fontSize: 90,
                scaleY: 50,
            }
        );

        root.add(sprite);

        return {
            value: yValue,
            sprite,
        };
    });


    /*
      Axis labels. The mathematical italic letters and subscript-like
      characters give a TeX-inspired appearance without an external
      MathJax/KaTeX dependency.
    */
    /*
      Axis labels are placed on both opposite sides of the grid.
    */
    const xAxisLabels = [
        makeGridTextSprite(
            '𝑥ₗₛᵣ [pc]',
            {
                fontSize: 82,
                scaleY: 56,
            }
        ),
        makeGridTextSprite(
            '𝑥ₗₛᵣ [pc]',
            {
                fontSize: 82,
                scaleY: 56,
            }
        ),
    ];

    const yAxisLabels = [
        makeGridTextSprite(
            '𝑦ₗₛᵣ [pc]',
            {
                fontSize: 82,
                scaleY: 56,
            }
        ),
        makeGridTextSprite(
            '𝑦ₗₛᵣ [pc]',
            {
                fontSize: 82,
                scaleY: 56,
            }
        ),
    ];

    for (const label of xAxisLabels) {
        root.add(label);
    }

    for (const label of yAxisLabels) {
        root.add(label);
    }

    /*
      Flat label below the negative-y side of the LSR coordinate square.
    */
    const lsrFrameLabel = makeFlatTextLabel(
        'LSR frame',
        {
            fontSize: 130,
            heightPc: 205,

            colour: 'rgba(235, 245, 255, 1.0)',
        }
    );

    /*
      Keep this farther below the square than the x_LSR axis label.
    */
    lsrFrameLabel.mesh.position.set(
        0.5 * (xMin + xMax),
        yMin - 260.0,
        zPlane - 2.0
    );

    root.add(lsrFrameLabel.mesh);

    root.visible = false;

    scene.add(root);

    /*
      ----------------------------------------------------------------
      Galactic z-axis
      ----------------------------------------------------------------

      The axis passes through:

          x = 0
          y = 0

      and spans the full z extent of the density-domain box.

      It uses independent material clones because its opacity depends
      on camera elevation, while the x-y frame must remain fully visible.

      The z axis is expressed in z_LSR coordinates.
      Because the Galactic-plane grid is physically at z = -20 pc:

          z_LSR = z_physical + 20 pc

      Thus:
          z_LSR = 0 corresponds to physical z = -20 pc.
    */
    const zAxis = createGalacticZAxis({
        zLsrMin: -500.0,
        zLsrMax: 500.0,

        /*
          Physical location of z_LSR = 0.
        */
        zLsrZeroPhysicalPc: zPlane,

        majorIntervalPc,
        minorIntervalPc,

        majorTickLengthPc,
        minorTickLengthPc,

        sourceGlowMaterial: spineGlowMaterial,
        sourceSpineMaterial: spineMaterial,
        sourceTickMaterial: tickMaterial,
    });

    root.add(zAxis.root);


    const grid = {
        root,

        xMin,
        xMax,
        yMin,
        yMax,

        zPlane,
        labelZ,

        planeCentre: new THREE.Vector3(
            0.5 * (xMin + xMax),
            0.5 * (yMin + yMax),
            zPlane
        ),

        lsrFrameLabel,

        gridMaterial,
        tickMaterial,

        spineGlowMaterial,
        spineMaterial,

        gridLines,
        minorTicks,
        majorTicks,

        frameGlow,
        frame,

        xTickLabels,
        yTickLabels,

        xAxisLabels,
        yAxisLabels,

        zAxis,

        labelSprites: [
            ...xTickLabels.map(
                entry => entry.sprite
            ),
            ...yTickLabels.map(
                entry => entry.sprite
            ),
            ...xAxisLabels,
            ...yAxisLabels,
        ],
    };

    /*
      Place labels correctly before the first render.
    */
    updateGalacticPlaneGridLabels(grid);

    return grid;
}


function updateGalacticPlaneGridStyle() {
    if (!galacticPlaneGrid) {
        return;
    }

    const frameOpacity = THREE.MathUtils.clamp(
        finiteNumber(params.gridFrameOpacity, 1.00),
        0.0,
        1.0
    );

    const lineOpacity = THREE.MathUtils.clamp(
        finiteNumber(params.gridLineOpacity, 0.30),
        0.0,
        1.0
    );

    galacticPlaneGrid.gridMaterial.opacity =
        lineOpacity;

    galacticPlaneGrid.tickMaterial.opacity =
        frameOpacity;

    galacticPlaneGrid.spineMaterial.opacity =
        frameOpacity;

    /*
      Keep the blue-white halo subtle but visible.
    */
    galacticPlaneGrid.spineGlowMaterial.opacity =
        frameOpacity * 0.22;

    for (const sprite of galacticPlaneGrid.labelSprites) {
        sprite.material.opacity = frameOpacity;
    }

    /*
      The z axis shares the grid-frame opacity, but also receives an
      additional camera-elevation fade.
    */
    updateGalacticZAxisStyle();
    updateLsrFrameLabelStyle();

}


function updateGalacticPlaneGridLabels(
    grid = galacticPlaneGrid
) {
    if (!grid) {
        return;
    }

    /*
      Labels are intentionally shown on both opposing sides.

      X-coordinate labels:
          yMin side and yMax side

      Y-coordinate labels:
          xMin side and xMax side
    */
    const tickOffset = 34.0;
    const titleOffset = 88.0;

    /*
      Major x-coordinate labels on the two y boundaries.
    */
    for (const entry of grid.xTickLabels) {
        /*
          Every x tick now owns two sprites: one for each side.
          This is initialized below if necessary.
        */
        if (!entry.sprites) {
            entry.sprites = [
                entry.sprite,
                makeGridTextSprite(
                    formatGridCoordinate(entry.value),
                    {
                        fontSize: 90,
                        scaleY: 50,
                    }
                ),
            ];

            grid.root.add(entry.sprites[1]);
            grid.labelSprites.push(entry.sprites[1]);
        }

        entry.sprites[0].position.set(
            entry.value,
            grid.yMin - tickOffset,
            grid.labelZ
        );

        entry.sprites[1].position.set(
            entry.value,
            grid.yMax + tickOffset,
            grid.labelZ
        );
    }

    /*
      Major y-coordinate labels on the two x boundaries.
    */
    for (const entry of grid.yTickLabels) {
        if (!entry.sprites) {
            entry.sprites = [
                entry.sprite,
                makeGridTextSprite(
                    formatGridCoordinate(entry.value),
                    {
                        fontSize: 90,
                        scaleY: 50,
                    }
                ),
            ];

            grid.root.add(entry.sprites[1]);
            grid.labelSprites.push(entry.sprites[1]);
        }

        entry.sprites[0].position.set(
            grid.xMin - tickOffset,
            entry.value,
            grid.labelZ
        );

        entry.sprites[1].position.set(
            grid.xMax + tickOffset,
            entry.value,
            grid.labelZ
        );
    }

    /*
      Axis titles appear only on the sides nearest the camera.
    */
    const xSideIsNearYMin =
        Math.abs(camera.position.y - grid.yMin)
        <= Math.abs(camera.position.y - grid.yMax);

    const ySideIsNearXMin =
        Math.abs(camera.position.x - grid.xMin)
        <= Math.abs(camera.position.x - grid.xMax);

    /*
      x_LSR: nearest y boundary.
    */
    grid.xAxisLabels[0].visible = xSideIsNearYMin;
    grid.xAxisLabels[1].visible = !xSideIsNearYMin;

    grid.xAxisLabels[0].position.set(
        0.5 * (grid.xMin + grid.xMax),
        grid.yMin - titleOffset,
        grid.labelZ
    );

    grid.xAxisLabels[1].position.set(
        0.5 * (grid.xMin + grid.xMax),
        grid.yMax + titleOffset,
        grid.labelZ
    );

    /*
      y_LSR: nearest x boundary.
    */
    grid.yAxisLabels[0].visible = ySideIsNearXMin;
    grid.yAxisLabels[1].visible = !ySideIsNearXMin;

    grid.yAxisLabels[0].position.set(
        grid.xMin - titleOffset,
        0.5 * (grid.yMin + grid.yMax),
        grid.labelZ
    );

    grid.yAxisLabels[1].position.set(
        grid.xMax + titleOffset,
        0.5 * (grid.yMin + grid.yMax),
        grid.labelZ
    );

}

function smoothVisibilityFactor(
    value,
    invisibleAt,
    fullyVisibleAt
) {
    const normalized = THREE.MathUtils.clamp(
        (value - invisibleAt)
        / Math.max(
            fullyVisibleAt - invisibleAt,
            1.0e-8
        ),
        0.0,
        1.0
    );

    return normalized
        * normalized
        * (
            3.0
            - 2.0 * normalized
        );
}


function cameraDistanceToLsrPlaneCentre() {
    if (!camera || !galacticPlaneGrid) {
        return 0.0;
    }

    return camera.position.distanceTo(
        galacticPlaneGrid.planeCentre
    );
}


function updateLsrFrameLabelStyle(
    grid = galacticPlaneGrid
) {
    if (!grid?.lsrFrameLabel || !camera) {
        return;
    }

    const distancePc =
        cameraDistanceToLsrPlaneCentre();

    const elevationDeg =
        cameraElevationAboveGalacticPlaneDeg();

    /*
      Invisible below 1.2 kpc; fully visible at >= 2.0 kpc.
    */
    const distanceFactor =
        smoothVisibilityFactor(
            distancePc,
            3000.0,
            4500.0
        );

    /*
      Invisible below 30 degrees inclination; fully visible at >= 45°.
    */
    const inclinationFactor =
        smoothVisibilityFactor(
            elevationDeg,
            10.0,
            25.0
        );

    const finalOpacity =
        distanceFactor
        * inclinationFactor;

    grid.lsrFrameLabel.material.opacity =
        finalOpacity;

    grid.lsrFrameLabel.mesh.visible =
        finalOpacity > 0.001;
}

/* -------------------------------------------------------------------------- */
/* GALACTIC Z AXIS                                                            */
/* -------------------------------------------------------------------------- */

function cameraElevationAboveGalacticPlaneDeg() {
    if (!camera || !controls) {
        return 90.0;
    }

    /*
      Use the viewing direction, rather than camera.position alone.

      This means the result remains meaningful if the camera target is
      moved in the future.

      viewVector points from the OrbitControls target toward the camera.
    */
    const viewVector = new THREE.Vector3()
        .subVectors(
            camera.position,
            controls.target
        );

    const horizontalDistance = Math.hypot(
        viewVector.x,
        viewVector.y
    );

    /*
      theta = 0 degrees:
          camera is level with the Galactic plane

      theta = 90 degrees:
          camera is directly above or below the Galactic plane
    */
    const elevationRadians = Math.atan2(
        Math.abs(viewVector.z),
        Math.max(horizontalDistance, 1.0e-8)
    );

    return THREE.MathUtils.radToDeg(
        elevationRadians
    );
}


function zAxisCameraOpacityFactor() {
    const elevationDeg =
        cameraElevationAboveGalacticPlaneDeg();

    /*
      Fully visible when viewed close to edge-on.
    */
    if (
        elevationDeg
        <= Z_AXIS_FULL_OPACITY_ANGLE_DEG
    ) {
        return 1.0;
    }

    /*
      Fully hidden at high elevation / near top-down view.
    */
    if (
        elevationDeg
        >= Z_AXIS_FADE_END_DEG
    ) {
        return 0.0;
    }

    /*
      Smoothstep fade:

      elevation = 15 deg -> 1
      elevation = 50 deg -> 0
    */
    const normalizedElevation =
        (
            elevationDeg
            - Z_AXIS_FULL_OPACITY_ANGLE_DEG
        )
        / (
            Z_AXIS_FADE_END_DEG
            - Z_AXIS_FULL_OPACITY_ANGLE_DEG
        );

    const smoothStep =
        normalizedElevation
        * normalizedElevation
        * (
            3.0
            - 2.0 * normalizedElevation
        );

    return 1.0 - smoothStep;
}


function createGalacticZAxis(options) {
    const {
        zLsrMin,
        zLsrMax,
        zLsrZeroPhysicalPc,

        majorIntervalPc,
        minorIntervalPc,

        majorTickLengthPc,
        minorTickLengthPc,

        sourceGlowMaterial,
        sourceSpineMaterial,
        sourceTickMaterial,
    } = options;

    const root = new THREE.Group();

    root.name = 'Galactic z-axis';

    /*
      z_LSR coordinate convention:

          z_LSR = z_physical - zLsrZeroPhysicalPc

      Since the xy grid lies at physical z = -20 pc:

          z_LSR = z_physical + 20 pc

      Hence:
          z_LSR = 0 -> z_physical = -20 pc.
    */
    const zLsrToPhysicalZ = (zLsr) => {
        return zLsr + zLsrZeroPhysicalPc;
    };

    const zPhysicalMin = zLsrToPhysicalZ(
        zLsrMin
    );

    const zPhysicalMax = zLsrToPhysicalZ(
        zLsrMax
    );


    /*
      ----------------------------------------------------------------
      Materials
      ----------------------------------------------------------------
    */

    const glowMaterial = sourceGlowMaterial.clone();

    glowMaterial.resolution.set(
        window.innerWidth,
        window.innerHeight
    );

    const spineMaterial = sourceSpineMaterial.clone();

    spineMaterial.resolution.set(
        window.innerWidth,
        window.innerHeight
    );

    const tickMaterial = sourceTickMaterial.clone();


    /*
      ----------------------------------------------------------------
      Main glowing z-axis
      ----------------------------------------------------------------
    */

    const axisPositions = new Float32Array([
        0.0, 0.0, zPhysicalMin,
        0.0, 0.0, zPhysicalMax,
    ]);

    const glowGeometry = new LineGeometry();
    glowGeometry.setPositions(axisPositions);

    const spineGeometry = new LineGeometry();
    spineGeometry.setPositions(axisPositions);

    const glowLine = new Line2(
        glowGeometry,
        glowMaterial
    );

    const spineLine = new Line2(
        spineGeometry,
        spineMaterial
    );

    glowLine.renderOrder = 2;
    spineLine.renderOrder = 3;

    glowLine.frustumCulled = false;
    spineLine.frustumCulled = false;

    root.add(glowLine);
    root.add(spineLine);


    /*
      ----------------------------------------------------------------
      Arrowhead at positive z_LSR
      ----------------------------------------------------------------

      Smaller and longer than the previous implementation.
    */

    const arrowHeightPc = 72.0;
    const arrowRadiusPc = 10.0;

    const arrowGeometry = new THREE.ConeGeometry(
        arrowRadiusPc,
        arrowHeightPc,
        20
    );

    const arrowMaterial = new THREE.MeshBasicMaterial({
        color: 0xf2fbff,

        transparent: true,
        opacity: 1.0,

        depthTest: true,
        depthWrite: false,

        toneMapped: false,
    });

    const arrowGlowGeometry = new THREE.ConeGeometry(
        arrowRadiusPc * 1.65,
        arrowHeightPc * 1.20,
        20
    );

    const arrowGlowMaterial = new THREE.MeshBasicMaterial({
        color: 0x8fd8ff,

        transparent: true,
        opacity: 0.20,

        depthTest: true,
        depthWrite: false,

        blending: THREE.AdditiveBlending,

        toneMapped: false,
    });

    const arrowGlow = new THREE.Mesh(
        arrowGlowGeometry,
        arrowGlowMaterial
    );

    const arrow = new THREE.Mesh(
        arrowGeometry,
        arrowMaterial
    );

    /*
      ConeGeometry points along local +Y by default.

      Rotate it so its tip points toward physical +Z.
    */
    arrowGlow.rotation.x = Math.PI * 0.5;
    arrow.rotation.x = Math.PI * 0.5;

    /*
      The axis ends at zPhysicalMax. The cone base touches that endpoint,
      while its tip extends upward.
    */
    arrowGlow.position.set(
        0.0,
        0.0,
        zPhysicalMax + 0.5 * arrowHeightPc
    );

    arrow.position.set(
        0.0,
        0.0,
        zPhysicalMax + 0.5 * arrowHeightPc
    );

    arrowGlow.renderOrder = 2;
    arrow.renderOrder = 4;

    root.add(arrowGlow);
    root.add(arrow);


    /*
      ----------------------------------------------------------------
      Tick-and-label group
      ----------------------------------------------------------------

      Local +X points toward the camera's horizontal direction.

      The group is rotated about the z axis in
      updateGalacticZAxisCameraFacing(), so ticks remain visible as the
      user orbits the scene.

      Each tick is centered on the z axis:

          -length / 2  ->  +length / 2

      rather than extending only toward +X.
    */

    const tickAndLabelGroup = new THREE.Group();

    tickAndLabelGroup.name =
        'Galactic z-axis ticks and labels';

    root.add(tickAndLabelGroup);

    const majorTickPositions = [];
    const minorTickPositions = [];

    const majorValues = [];

    const firstMinorZ =
        Math.ceil(zLsrMin / minorIntervalPc)
        * minorIntervalPc;

    function isMajorZCoordinate(value) {
        const scaled = value / majorIntervalPc;

        return Math.abs(
            scaled - Math.round(scaled)
        ) < 1.0e-6;
    }

    for (
        let zLsr = firstMinorZ;
        zLsr <= zLsrMax + 1.0e-6;
        zLsr += minorIntervalPc
    ) {
        /*
          Do not duplicate the endpoint/spine geometry with a tick.
        */
        if (
            zLsr <= zLsrMin + 1.0e-6
            || zLsr >= zLsrMax - 1.0e-6
        ) {
            continue;
        }

        const isMajor = isMajorZCoordinate(
            zLsr
        );

        const tickLength = isMajor
            ? majorTickLengthPc
            : minorTickLengthPc;

        const targetPositions = isMajor
            ? majorTickPositions
            : minorTickPositions;

        const zPhysical = zLsrToPhysicalZ(
            zLsr
        );

        /*
          Centered tick: -x to +x.
        */
        targetPositions.push(
            -0.5 * tickLength,
            0.0,
            zPhysical,

            +0.5 * tickLength,
            0.0,
            zPhysical
        );

        if (isMajor) {
            majorValues.push(zLsr);
        }
    }

    const minorTicks = makeLineSegmentsObject(
        minorTickPositions,
        tickMaterial,
        3
    );

    const majorTicks = makeLineSegmentsObject(
        majorTickPositions,
        tickMaterial,
        4
    );

    tickAndLabelGroup.add(minorTicks);
    tickAndLabelGroup.add(majorTicks);


    /*
      ----------------------------------------------------------------
      Major tick labels
      ----------------------------------------------------------------

      Local +X is rotated to face the camera, keeping labels visible.
    */

    const tickLabels = majorValues.map((zLsr) => {
        const sprite = makeGridTextSprite(
            formatGridCoordinate(zLsr),
            {
                fontSize: 62,
                scaleY: 50,
            }
        );

        const zPhysical = zLsrToPhysicalZ(
            zLsr
        );

        sprite.position.set(
            0.5 * majorTickLengthPc + 35.0,
            0.0,
            zPhysical
        );

        tickAndLabelGroup.add(sprite);

        return {
            value: zLsr,
            sprite,
        };
    });


    /*
      ----------------------------------------------------------------
      z_LSR axis title
      ----------------------------------------------------------------
    */

    const axisLabel = makeGridTextSprite(
        '𝑧ₗₛᵣ [pc]',
        {
            fontSize: 82,
            scaleY: 56,
        }
    );

    axisLabel.position.set(
        arrowRadiusPc + 60.0,
        0.0,
        zPhysicalMax + arrowHeightPc + 36.0
    );

    tickAndLabelGroup.add(axisLabel);


    return {
        root,

        zLsrMin,
        zLsrMax,
        zLsrZeroPhysicalPc,

        zPhysicalMin,
        zPhysicalMax,

        glowMaterial,
        spineMaterial,
        tickMaterial,

        glowLine,
        spineLine,

        arrowGlowMaterial,
        arrowMaterial,

        arrowGlow,
        arrow,

        tickAndLabelGroup,

        minorTicks,
        majorTicks,

        tickLabels,
        axisLabel,
    };
}

function updateGalacticZAxisCameraFacing(
    grid = galacticPlaneGrid
) {
    if (!grid?.zAxis || !camera) {
        return;
    }

    const zAxis = grid.zAxis;

    /*
      Horizontal camera direction relative to the z axis.

      The tick-and-label group is rotated 90 degrees away from the
      camera azimuth. Therefore:

      - tick marks are viewed broadside rather than end-on;
      - tick labels remain visibly offset from the z axis;
      - labels do not collapse onto the central vertical line.
    */
    const horizontalDistance = Math.hypot(
        camera.position.x,
        camera.position.y
    );

    if (horizontalDistance < 1.0e-8) {
        return;
    }

    const cameraAzimuth = Math.atan2(
        camera.position.y,
        camera.position.x
    );

    /*
      Local +X becomes perpendicular to the horizontal camera direction.
      This keeps the tick marks visibly extended in the rendered view.
    */
    zAxis.tickAndLabelGroup.rotation.z =
        cameraAzimuth + Math.PI * 0.5;
}

function updateGalacticZAxisStyle(
    grid = galacticPlaneGrid
) {
    if (!grid?.zAxis) {
        return;
    }

    const zAxis = grid.zAxis;

    /*
      Rotate ticks, tick labels, and the z-axis title toward the camera.
    */
    updateGalacticZAxisCameraFacing(grid);

    const frameOpacity = THREE.MathUtils.clamp(
        finiteNumber(params.gridFrameOpacity, 1.0),
        0.0,
        1.0
    );

    /*
      This factor depends on camera elevation above the Galactic plane.
    */
    const cameraFade =
        zAxisCameraOpacityFactor();

    const finalOpacity =
        frameOpacity
        * cameraFade;

    /*
      Hide the entire group once opacity is effectively zero.

      The parent grid itself is still controlled independently through
      params.showGalacticPlaneGrid.
    */
    zAxis.root.visible = finalOpacity > 0.001;

    zAxis.spineMaterial.opacity =
        finalOpacity;

    zAxis.glowMaterial.opacity =
        finalOpacity * 0.22;

    zAxis.tickMaterial.opacity =
        finalOpacity;

    zAxis.arrowMaterial.opacity =
        finalOpacity;

    zAxis.arrowGlowMaterial.opacity =
        finalOpacity * 0.20;

    for (const entry of zAxis.tickLabels) {
        entry.sprite.material.opacity =
            finalOpacity;
    }

    zAxis.axisLabel.material.opacity =
        finalOpacity;
}

/* -------------------------------------------------------------------------- */
/* SOLAR CIRCLE                                                               */
/* -------------------------------------------------------------------------- */

function solarCircleAngularSpeedRadPerMyr() {
    /*
      Circular speed in pc/Myr.
    */
    const circularSpeedPcPerMyr =
        SOLAR_CIRCLE_SPEED_KM_S
        * KM_S_TO_PC_MYR;

    /*
      omega = V / R.
    */
    return (
        circularSpeedPcPerMyr
        / SOLAR_CIRCLE_RADIUS_PC
    );
}


function solarCircleCentreAtTime(timeMyr) {
    /*
      This reproduces the convention used in your Python script:

          gc_vec = [cos(omega * t), -sin(omega * t)]

      The local LSR volume remains at the origin. The apparent Galactic
      centre rotates around it as time changes.
    */
    const angularPosition =
        solarCircleAngularSpeedRadPerMyr()
        * timeMyr;

    return new THREE.Vector2(
        SOLAR_CIRCLE_RADIUS_PC
        * Math.cos(angularPosition),

        -SOLAR_CIRCLE_RADIUS_PC
        * Math.sin(angularPosition)
    );
}


function createSolarCircleLayer() {
    /*
      Solar circle, Galactic-centre marker, and Galactic-centre label
      all lie on the same local LSR plane.
    */
    const zPlane = -20.0;

    const numberOfCircleSegments = 360;

    const root = new THREE.Group();

    root.name = 'Solar circle and Galactic centre guide';


    /*
      ----------------------------------------------------------------
      Solar circle
      ----------------------------------------------------------------
    */

    const circleGeometry = new LineGeometry();

    circleGeometry.setPositions(
        new Float32Array([
            0, 0, zPlane,
            0, 0, zPlane,
        ])
    );

    const circleMaterial = new LineMaterial({
        color: 0xa9b2ba,

        linewidth: SOLAR_CIRCLE_LINE_WIDTH_PX,

        transparent: true,
        opacity: SOLAR_CIRCLE_OPACITY,

        depthTest: true,
        depthWrite: false,

        worldUnits: false,
        toneMapped: false,
    });

    circleMaterial.resolution.set(
        window.innerWidth,
        window.innerHeight
    );

    const circleLine = new Line2(
        circleGeometry,
        circleMaterial
    );

    circleLine.name = 'Solar circle';

    circleLine.renderOrder = 1;
    circleLine.frustumCulled = false;

    root.add(circleLine);


    /*
      ----------------------------------------------------------------
      Inward Solar-circle ticks
      ----------------------------------------------------------------
    */

    const tickGeometry = new THREE.BufferGeometry();

    tickGeometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(
            new Float32Array(0),
            3
        )
    );

    const tickMaterial = new THREE.LineBasicMaterial({
        color: 0xa9b2ba,

        transparent: true,
        opacity: SOLAR_CIRCLE_OPACITY,

        depthTest: true,
        depthWrite: false,

        toneMapped: false,
    });

    const tickLines = new THREE.LineSegments(
        tickGeometry,
        tickMaterial
    );

    tickLines.name = 'Solar circle inward ticks';

    tickLines.renderOrder = 1;
    tickLines.frustumCulled = false;

    root.add(tickLines);

    /*
      ----------------------------------------------------------------
      Galactic-centre radial reference line
      ----------------------------------------------------------------

      This connects the local LSR origin to the Galactic centre.
    */
    const galacticRadiusGeometry = new LineGeometry();

    galacticRadiusGeometry.setPositions(
        new Float32Array([
            0.0, 0.0, zPlane,
            0.0, 0.0, zPlane,
        ])
    );

    const galacticRadiusMaterial = new LineMaterial({
        color: 0xb9c3cd,

        linewidth: 1.0,

        transparent: true,
        opacity: 0.0,

        depthTest: true,
        depthWrite: false,

        worldUnits: false,
        toneMapped: false,
    });

    galacticRadiusMaterial.resolution.set(
        window.innerWidth,
        window.innerHeight
    );

    const galacticRadiusLine = new Line2(
        galacticRadiusGeometry,
        galacticRadiusMaterial
    );

    galacticRadiusLine.name =
        'Solar Galactocentric radius';

    galacticRadiusLine.renderOrder = 1;
    galacticRadiusLine.frustumCulled = false;

    root.add(galacticRadiusLine);


    /*
      Same font-size and world scale as the Galactic-center label.
    */
    /*
      A real flat text plane rather than a Sprite.

      Unlike a billboard Sprite, this can act as a flag whose base stays
      parallel to the Solar Galactocentric-radius line while rotating to
      face the camera as much as possible.
    */
    const galacticRadiusLabel = makeFlatTextLabel(
        'Galactocentric radius = 8.12 kpc',
        {
            fontSize: GALACTIC_RADIUS_LABEL_FONT_SIZE_PX,

            heightPc: GALACTIC_RADIUS_LABEL_HEIGHT_PC,

            colour: 'rgba(220, 230, 240, 0.95)',
        }
    );

    galacticRadiusLabel.material.opacity = 0.0;

    galacticRadiusLabel.mesh.renderOrder =
        REFERENCE_RENDER_ORDER;

    root.add(galacticRadiusLabel.mesh);

    /*
      ----------------------------------------------------------------
      Moving Galactic-centre glow marker
      ----------------------------------------------------------------
    */

    const galacticCentreGeometry =
        new THREE.BufferGeometry();

    const galacticCentrePositionAttribute =
        new THREE.BufferAttribute(
            new Float32Array([
                0.0,
                0.0,
                zPlane,
            ]),
            3
        );

    galacticCentrePositionAttribute.setUsage(
        THREE.DynamicDrawUsage
    );

    galacticCentreGeometry.setAttribute(
        'position',
        galacticCentrePositionAttribute
    );

    const gl = renderer.getContext();

    const pointSizeRange = gl.getParameter(
        gl.ALIASED_POINT_SIZE_RANGE
    );

    const galacticCentreMaterial =
        new THREE.ShaderMaterial({
            uniforms: {
                uDiameterPc: {
                    value:
                        GALACTIC_CENTRE_GLOW_DIAMETER_PC,
                },

                uProjectionScale: {
                    value: 1.0,
                },

                uMaxPointSize: {
                    value: Number(
                        pointSizeRange[1]
                    ),
                },
            },

            vertexShader:
                GALACTIC_CENTRE_VERTEX_SHADER,

            fragmentShader:
                GALACTIC_CENTRE_FRAGMENT_SHADER,

            transparent: true,

            depthTest: true,
            depthWrite: false,

            blending: THREE.AdditiveBlending,

            toneMapped: false,
        });

    const galacticCentrePoints = new THREE.Points(
        galacticCentreGeometry,
        galacticCentreMaterial
    );

    galacticCentrePoints.name = 'Galactic center';

    galacticCentrePoints.frustumCulled = false;
    galacticCentrePoints.renderOrder = 2;

    root.add(galacticCentrePoints);


    /*
      ----------------------------------------------------------------
      Galactic-centre label
      ----------------------------------------------------------------

      Canvas sprite text is used, as for the grid labels. The larger
      canvas font and physical sprite scale preserve legibility.
    */

    const galacticCentreLabel = makeGridTextSprite(
        'Galactic center',
        {
            fontSize: 130,
            fontFamily: 'Georgia, Times New Roman, serif',

            colour: 'rgba(255, 222, 228, 1.0)',

            scaleY: 250,
        }
    );

    /*
      Its exact x/y position is updated whenever the Solar circle moves.
      The offset keeps the label outside the red glow marker.
    */
    galacticCentreLabel.renderOrder = 5;

    root.add(galacticCentreLabel);


    scene.add(root);

    const layer = {
        root,

        zPlane,

        numberOfCircleSegments,

        circleGeometry,
        circleMaterial,
        circleLine,

        tickGeometry,
        tickMaterial,
        tickLines,

        galacticRadiusGeometry,
        galacticRadiusMaterial,
        galacticRadiusLine,
        galacticRadiusLabel,

        galacticCentreGeometry,
        galacticCentrePositionAttribute,
        galacticCentreMaterial,
        galacticCentrePoints,
        galacticCentreLabel,

        lastTimeMyr: Number.NaN,
    };

    updateSolarCircle(
        0.0,
        layer
    );

    updateGalacticCentreProjectionScale(
        layer
    );

    return layer;
}


function updateSolarCircle(
    timeMyr,
    layer = solarCircleLayer
) {
    if (!layer) {
        return;
    }

    if (
        Math.abs(
            timeMyr - layer.lastTimeMyr
        ) < 1.0e-8
    ) {
        return;
    }

    const centreAtTime = solarCircleCentreAtTime(
        timeMyr
    );

    const centreX = centreAtTime.x;
    const centreY = centreAtTime.y;

    const radius = SOLAR_CIRCLE_RADIUS_PC;
    const zPlane = layer.zPlane;

    /*
      ----------------------------------------------------------------
      Local origin -> Galactic centre reference line
      ----------------------------------------------------------------
    */
    const oldGalacticRadiusGeometry =
        layer.galacticRadiusLine.geometry;

    const newGalacticRadiusGeometry =
        new LineGeometry();

    newGalacticRadiusGeometry.setPositions(
        new Float32Array([
            0.0,
            0.0,
            zPlane,

            centreX,
            centreY,
            zPlane,
        ])
    );

    layer.galacticRadiusGeometry =
        newGalacticRadiusGeometry;

    layer.galacticRadiusLine.geometry =
        newGalacticRadiusGeometry;

    oldGalacticRadiusGeometry.dispose();

    /*
      ----------------------------------------------------------------
      Solar circle
      ----------------------------------------------------------------
    */

    const circlePositions = new Float32Array(
        (layer.numberOfCircleSegments + 1) * 3
    );

    for (
        let pointIndex = 0;
        pointIndex <= layer.numberOfCircleSegments;
        pointIndex++
    ) {
        const theta =
            (
                pointIndex
                / layer.numberOfCircleSegments
            )
            * Math.PI
            * 2.0;

        const destinationIndex = pointIndex * 3;

        circlePositions[destinationIndex + 0] =
            centreX
            + radius * Math.cos(theta);

        circlePositions[destinationIndex + 1] =
            centreY
            + radius * Math.sin(theta);

        circlePositions[destinationIndex + 2] =
            zPlane;
    }

    const oldCircleGeometry =
        layer.circleLine.geometry;

    const newCircleGeometry = new LineGeometry();

    newCircleGeometry.setPositions(
        circlePositions
    );

    layer.circleGeometry = newCircleGeometry;
    layer.circleLine.geometry = newCircleGeometry;

    oldCircleGeometry.dispose();

    layer.circleLine.computeLineDistances();


    /*
      ----------------------------------------------------------------
      Solar-circle inward ticks
      ----------------------------------------------------------------
    */

    const numberOfTicks = Math.round(
        360.0 / SOLAR_CIRCLE_TICK_INTERVAL_DEG
    );

    const tickPositions = new Float32Array(
        numberOfTicks * 2 * 3
    );

    for (
        let tickIndex = 0;
        tickIndex < numberOfTicks;
        tickIndex++
    ) {
        const theta =
            tickIndex
            * THREE.MathUtils.degToRad(
                SOLAR_CIRCLE_TICK_INTERVAL_DEG
            );

        const radialX = Math.cos(theta);
        const radialY = Math.sin(theta);

        const circleX =
            centreX
            + radius * radialX;

        const circleY =
            centreY
            + radius * radialY;

        const innerX =
            circleX
            - SOLAR_CIRCLE_TICK_LENGTH_PC
            * radialX;

        const innerY =
            circleY
            - SOLAR_CIRCLE_TICK_LENGTH_PC
            * radialY;

        const destinationIndex = tickIndex * 6;

        tickPositions[destinationIndex + 0] =
            circleX;

        tickPositions[destinationIndex + 1] =
            circleY;

        tickPositions[destinationIndex + 2] =
            zPlane;

        tickPositions[destinationIndex + 3] =
            innerX;

        tickPositions[destinationIndex + 4] =
            innerY;

        tickPositions[destinationIndex + 5] =
            zPlane;
    }

    const oldTickGeometry =
        layer.tickLines.geometry;

    const newTickGeometry =
        new THREE.BufferGeometry();

    newTickGeometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(
            tickPositions,
            3
        )
    );

    layer.tickGeometry = newTickGeometry;
    layer.tickLines.geometry = newTickGeometry;

    oldTickGeometry.dispose();


    /*
      ----------------------------------------------------------------
      Galactic-centre marker and label
      ----------------------------------------------------------------

      The Galactic centre is the centre of the Solar circle.
    */

    layer.galacticCentrePositionAttribute.array[0] =
        centreX;

    layer.galacticCentrePositionAttribute.array[1] =
        centreY;

    layer.galacticCentrePositionAttribute.array[2] =
        zPlane;

    layer.galacticCentrePositionAttribute.needsUpdate =
        true;

    updateGalacticCentreLabelPosition(
        centreX,
        centreY,
        zPlane,
        layer
    );

    layer.lastTimeMyr = timeMyr;
}

function updateGalacticCentreLabelPosition(
    centreX,
    centreY,
    centreZ,
    layer = solarCircleLayer
) {
    if (!layer || !camera) {
        return;
    }

    /*
      The camera's local Y axis is its screen-up direction in world space.

      This gives the label a position visually above the Galactic-centre
      point regardless of the current OrbitControls orientation.
    */
    const cameraUpWorld = new THREE.Vector3(
        0.0,
        1.0,
        0.0
    ).transformDirection(
        camera.matrixWorld
    );

    /*
      This is a physical offset in pc. Increase if the label overlaps
      the crimson glow at close zoom.
    */
    const labelOffsetPc = 200.0;

    layer.galacticCentreLabel.position.set(
        centreX,
        centreY,
        centreZ
    );

    layer.galacticCentreLabel.position.addScaledVector(
        cameraUpWorld,
        labelOffsetPc
    );
}

function updateGalacticRadiusLabelPosition(
    centreX,
    centreY,
    zPlane,
    layer = solarCircleLayer
) {
    if (!layer || !camera) {
        return;
    }

    camera.updateMatrixWorld(true);

    /*
      The Solar-radius line runs from the local LSR origin to the
      Galactic centre.
    */
    const lineDirection = new THREE.Vector3(
        centreX,
        centreY,
        0.0
    );

    const lineLength = lineDirection.length();

    if (lineLength < 1.0e-8) {
        return;
    }

    lineDirection.divideScalar(lineLength);

    /*
      Attachment point: midpoint of the Solar-radius line.

      This is approximately 4.06 kpc from the Galactic centre and from
      the local LSR origin.
    */
    const attachmentPoint = new THREE.Vector3(
        0.5 * centreX,
        0.5 * centreY,
        zPlane
    );

    /*
      Camera direction, measured from the attachment point.
    */
    const directionToCamera = new THREE.Vector3()
        .subVectors(
            camera.position,
            attachmentPoint
        )
        .normalize();

    /*
      The label must retain one in-plane direction parallel to the
      Solar-radius line. Therefore, use the component of the camera
      direction perpendicular to that line as the plane normal.

      This creates the most camera-facing possible flag plane subject
      to the "attached to the line" constraint.
    */
    const labelNormal = directionToCamera
        .clone()
        .addScaledVector(
            lineDirection,
            -directionToCamera.dot(
                lineDirection
            )
        );

    /*
      Degenerate case: camera is looking almost exactly along the line.
      Use the camera up direction as a stable fallback.
    */
    if (labelNormal.lengthSq() < 1.0e-10) {
        labelNormal.set(
            0.0,
            1.0,
            0.0
        ).transformDirection(
            camera.matrixWorld
        );

        labelNormal.addScaledVector(
            lineDirection,
            -labelNormal.dot(
                lineDirection
            )
        );
    }

    labelNormal.normalize();

    /*
      Flat text plane basis:

          local X = along the Solar-radius line
          local Y = outward from the line to the label body
          local Z = plane normal, oriented toward the camera

      The plane's lower edge is therefore parallel to and conceptually
      attached to the radius line.
    */
    let localX = lineDirection.clone();

    /*
      Keep text approximately upright from the viewer's perspective.

      Flipping local X by 180 degrees still keeps it parallel to the
      line, but avoids unnecessarily upside-down text.
    */
    const cameraRight = new THREE.Vector3(
        1.0,
        0.0,
        0.0
    ).transformDirection(
        camera.matrixWorld
    );

    if (localX.dot(cameraRight) < 0.0) {
        localX.negate();
    }

    const localY = new THREE.Vector3()
        .crossVectors(
            labelNormal,
            localX
        )
        .normalize();

    /*
      Ensure the plane normal stays camera-facing after the local-X
      possible flip.
    */
    const correctedNormal = new THREE.Vector3()
        .crossVectors(
            localX,
            localY
        )
        .normalize();

    /*
      Build the orientation matrix.

      PlaneGeometry lies in local XY and has local +Z as its normal.
      Therefore its world basis is:

          local X -> localX
          local Y -> localY
          local Z -> correctedNormal
    */
    const orientationMatrix = new THREE.Matrix4().makeBasis(
        localX,
        localY,
        correctedNormal
    );

    layer.galacticRadiusLabel.mesh.quaternion
        .setFromRotationMatrix(
            orientationMatrix
        );

    /*
      Place the label's centre beyond the line.

      Its lower edge is separated from the line by:
          GALACTIC_RADIUS_LABEL_LINE_SEPARATION_PC

      Since PlaneGeometry is centred around its local origin, move its
      centre by half its physical height plus the requested separation.
    */
    const labelCentreOffset =
        GALACTIC_RADIUS_LABEL_LINE_SEPARATION_PC
        + 0.5
        * GALACTIC_RADIUS_LABEL_HEIGHT_PC;

    layer.galacticRadiusLabel.mesh.position
        .copy(attachmentPoint)
        .addScaledVector(
            localY,
            labelCentreOffset
        );
}

function updateGalacticCentreProjectionScale(
    layer = solarCircleLayer
) {
    if (!layer || !camera) {
        return;
    }

    const drawingBufferSize =
        renderer.getDrawingBufferSize(
            new THREE.Vector2()
        );

    const verticalFovRadians =
        THREE.MathUtils.degToRad(
            camera.fov
        );

    const projectionScale =
        drawingBufferSize.y
        / (
            2.0
            * Math.tan(
                verticalFovRadians * 0.5
            )
        );

    layer.galacticCentreMaterial
        .uniforms
        .uProjectionScale
        .value = projectionScale;
}

function updateSolarCircleStyle() {
    if (!solarCircleLayer || !clusterLayer) {
        return;
    }

    /*
      The Show Galactic references checkbox controls every child:
      Solar circle, ticks, Galactic centre, labels, and radius line.
    */
    solarCircleLayer.root.visible = Boolean(
        params.showSolarCircle
    );

    solarCircleLayer.circleMaterial.opacity =
        SOLAR_CIRCLE_OPACITY;

    solarCircleLayer.tickMaterial.opacity =
        SOLAR_CIRCLE_OPACITY;

    const selectedTimeMyr =
        clusterLayer.timesMyr[
            clusterLayer.frameIndex
        ];

    const centreAtTime = solarCircleCentreAtTime(
        selectedTimeMyr
    );

    /*
      Keep the Galactic-center label screen-above its crimson marker.
    */
    updateGalacticCentreLabelPosition(
        centreAtTime.x,
        centreAtTime.y,
        solarCircleLayer.zPlane
    );

    /*
      Keep R = 8.12 kpc offset from and parallel to the radial line.
    */
    updateGalacticRadiusLabelPosition(
        centreAtTime.x,
        centreAtTime.y,
        solarCircleLayer.zPlane
    );

    /*
      Radius line and its label:

      distance < 2 kpc -> invisible
      distance > 3 kpc -> fully visible
      between          -> smooth transition
    */
    const cameraDistancePc =
        cameraDistanceToLsrPlaneCentre();

    const radiusReferenceOpacity =
        smoothVisibilityFactor(
            cameraDistancePc,
            GALACTIC_RADIUS_REFERENCE_FADE_START_PC,
            GALACTIC_RADIUS_REFERENCE_FADE_END_PC
        );

    solarCircleLayer.galacticRadiusMaterial.opacity =
        GALACTIC_RADIUS_LINE_NOMINAL_OPACITY
        * radiusReferenceOpacity;

    solarCircleLayer.galacticRadiusLabel.material.opacity =
        GALACTIC_RADIUS_LABEL_NOMINAL_OPACITY
        * radiusReferenceOpacity;

    solarCircleLayer.galacticRadiusLabel.mesh.visible =
        radiusReferenceOpacity > 0.001;

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

async function loadText(url) {
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(
            `Could not load ${url}: HTTP ${response.status}`
        );
    }

    return response.text();
}

async function loadArrayBuffer(url) {
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Could not load ${url}: HTTP ${response.status}`);
    }

    return response.arrayBuffer();
}

function updateCameraViewOffset() {
    if (!camera) {
        return;
    }

    /*
      setViewOffset() changes the virtual camera viewport.

      A positive vertical offset shifts the rendered scene upward in the
      visible canvas, providing more room above the lower time slider.

      The full canvas dimensions are retained, so the camera aspect ratio
      and OrbitControls behavior remain unchanged.
    */
    camera.setViewOffset(
        window.innerWidth,
        window.innerHeight,

        0,
        CAMERA_VERTICAL_VIEW_OFFSET_PX,

        window.innerWidth,
        window.innerHeight
    );

    camera.updateProjectionMatrix();
}

function returnToPresentTimeSmoothly() {
    if (!clusterLayer) {
        return;
    }

    /*
      Cancel only an existing time-return animation.

      This does not affect a simultaneous camera-reset animation.
    */
    if (timeReturnAnimationFrame !== null) {
        cancelAnimationFrame(
            timeReturnAnimationFrame
        );

        timeReturnAnimationFrame = null;
    }

    const startTimeMyr = finiteNumber(
        params.clusterTime,
        clusterLayer.timesMyr[
            clusterLayer.zeroTimeIndex
        ]
    );

    const presentTimeMyr = clusterLayer.timesMyr[
        clusterLayer.zeroTimeIndex
    ];

    /*
      Already at the present epoch.
    */
    if (
        Math.abs(
            startTimeMyr - presentTimeMyr
        ) < 1.0e-8
    ) {
        return;
    }

    const durationMs = 2000.0;
    const animationStart = performance.now();

    function smoothStep(value) {
        return value * value * (
            3.0 - 2.0 * value
        );
    }

    function animateTimeReturn(now) {
        const rawProgress = THREE.MathUtils.clamp(
            (now - animationStart) / durationMs,
            0.0,
            1.0
        );

        const easedProgress = smoothStep(
            rawProgress
        );

        /*
          The trajectory dataset contains discrete 1-Myr snapshots, so
          the viewer advances through those precomputed epochs gradually
          over the 2-second interval.
        */
        params.clusterTime = THREE.MathUtils.lerp(
            startTimeMyr,
            presentTimeMyr,
            easedProgress
        );

        updateExternalTimeSlider();

        requestRender();

        if (rawProgress < 1.0) {
            timeReturnAnimationFrame =
                requestAnimationFrame(
                    animateTimeReturn
                );
        } else {
            /*
              End exactly at t = 0 Myr.
            */
            params.clusterTime = presentTimeMyr;

            setClusterFrameFromTime(
                presentTimeMyr
            );

            updateExternalTimeSlider();

            timeReturnAnimationFrame = null;

            requestRender();
        }
    }

    timeReturnAnimationFrame =
        requestAnimationFrame(
            animateTimeReturn
        );
}

function isTextEditingElement(element) {
    if (!element) {
        return false;
    }

    if (element.isContentEditable) {
        return true;
    }

    if (
        element instanceof HTMLTextAreaElement
        || element instanceof HTMLSelectElement
    ) {
        return true;
    }

    if (element instanceof HTMLInputElement) {
        /*
          Allow R and T while a range slider, checkbox, or button has
          focus. Still avoid hijacking keys while the user edits a
          lil-gui numerical/text input.
        */
        const nonTextInputTypes = new Set([
            'range',
            'checkbox',
            'radio',
            'button',
            'submit',
            'reset',
            'color',
            'file',
        ]);

        return !nonTextInputTypes.has(
            element.type
        );
    }

    return false;
}


function onKeyDown(event) {
    /*
      Do not repeatedly restart a two-second animation while the key is
      held down.
    */
    if (event.repeat) {
        return;
    }

    /*
      Preserve browser shortcuts such as Ctrl+R, Cmd+R, etc.
    */
    if (
        event.ctrlKey
        || event.metaKey
        || event.altKey
    ) {
        return;
    }

    if (
        isTextEditingElement(
            document.activeElement
        )
    ) {
        return;
    }

    if (event.code === 'KeyR') {
        event.preventDefault();
        event.stopPropagation();

        params?.resetView?.();

        return;
    }

    if (event.code === 'KeyT') {
        event.preventDefault();
        event.stopPropagation();

        returnToPresentTimeSmoothly();
    }
}

function onResize() {
    renderer.setSize(
        window.innerWidth,
        window.innerHeight
    );

    if (camera) {
        camera.aspect =
            window.innerWidth / window.innerHeight;
        updateCameraViewOffset();
    }

    if (gouldBeltMaterial) {
        gouldBeltMaterial.resolution.set(
            window.innerWidth,
            window.innerHeight
        );
    }

    if (radcliffeWaveMaterial) {
        radcliffeWaveMaterial.resolution.set(
            window.innerWidth,
            window.innerHeight
        );
    }

    if (solarCircleLayer) {
        solarCircleLayer.circleMaterial.resolution.set(
            window.innerWidth,
            window.innerHeight
        );

        solarCircleLayer.galacticRadiusMaterial.resolution.set(
            window.innerWidth,
            window.innerHeight
        );

        updateGalacticCentreProjectionScale();
    }

    if (sunLayer) {
        updateSunProjectionScale();
    }

    if (galacticPlaneGrid) {
        galacticPlaneGrid.spineGlowMaterial.resolution.set(
            window.innerWidth,
            window.innerHeight
        );

        galacticPlaneGrid.spineMaterial.resolution.set(
            window.innerWidth,
            window.innerHeight
        );

        /*
          z-axis LineMaterial objects have independent material instances.
        */
        if (galacticPlaneGrid.zAxis) {
            galacticPlaneGrid.zAxis.glowMaterial.resolution.set(
                window.innerWidth,
                window.innerHeight
            );

            galacticPlaneGrid.zAxis.spineMaterial.resolution.set(
                window.innerWidth,
                window.innerHeight
            );
        }

        updateGalacticPlaneGridLabels();
        updateGalacticZAxisCameraFacing();
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
