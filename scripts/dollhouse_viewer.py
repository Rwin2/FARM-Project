#!/usr/bin/env python3
"""Serve an interactive dollhouse viewer for a habitat scene mesh.

Stages the scene .glb next to a small three.js page (orbit / zoom / pan,
materials rendered unlit because scan textures carry baked lighting) and
serves it over HTTP. Runs on the host; the browser loads three.js from a
CDN, so it needs internet on the client side only.

Usage::

    python scripts/dollhouse_viewer.py --scene /path/to/scene.glb --port 8092
"""

from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import tempfile
from pathlib import Path

_PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title} — dollhouse viewer</title>
  <style>
    html, body {{ margin: 0; height: 100%; background: #111; }}
    #info {{ position: fixed; top: 8px; left: 12px; color: #ccc; font: 13px sans-serif; z-index: 1; }}
  </style>
  <script type="importmap">
    {{"imports": {{
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
    }}}}
  </script>
</head>
<body>
<div id="info">{title} — drag to orbit, scroll to zoom, right-drag to pan</div>
<script type="module">
  import * as THREE from 'three';
  import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
  import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x181818);
  const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(devicePixelRatio);
  document.body.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  new GLTFLoader().load('{mesh}', (gltf) => {{
    // Scan meshes have lighting baked into their textures — render unlit.
    gltf.scene.traverse((o) => {{
      if (o.isMesh && o.material) {{
        o.material = new THREE.MeshBasicMaterial({{
          map: o.material.map ?? null,
          color: o.material.color ?? 0xffffff,
          vertexColors: o.geometry.hasAttribute('color'),
        }});
      }}
    }});
    scene.add(gltf.scene);
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const c = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    camera.position.set(c.x + size * 0.45, c.y + size * 0.5, c.z + size * 0.45);
    controls.target.copy(c);
    controls.update();
  }});

  addEventListener('resize', () => {{
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  }});
  renderer.setAnimationLoop(() => {{ controls.update(); renderer.render(scene, camera); }});
</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True, help="Path to the scene .glb.")
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--bind", type=str, default="127.0.0.1")
    args = p.parse_args()

    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        raise SystemExit(f"scene not found: {scene}")

    with tempfile.TemporaryDirectory(prefix="dollhouse_") as tmp:
        root = Path(tmp)
        shutil.copy2(scene, root / scene.name)
        (root / "index.html").write_text(_PAGE.format(title=scene.stem, mesh=scene.name))
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
        with http.server.ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
            print(f"Serving {scene.name} on http://localhost:{args.port} — Ctrl+C to stop.")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("Bye.")


if __name__ == "__main__":
    main()
