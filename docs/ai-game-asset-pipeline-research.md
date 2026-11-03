# Deconstructing the AI Game Asset Pipeline: High-Fidelity 3D Generation, PBR Synthesis, and Physics-Assisted Kinematics

The proliferation of high-quality, AI-assisted independent games is frequently misunderstood as the output of autonomous, end-to-end generative tools. In practice, direct prompting of complete 3D assets invariably produces unusable results characterised by non-manifold triangular meshes, baked-in environmental illumination, distorted texture projections, and rubbery, floating character animations.

High-performing practitioners treat generative artificial intelligence as an array of discrete, specialised micro-services embedded within established computer graphics workflows. By decoupling production into controlled phases (concept orthography, geometric synthesis, photometric de-lighting, algorithmic retopology, and physics-informed character rigging), developers achieve visual and technical fidelity comparable to conventional studio assets. This ecosystem is increasingly orchestrated through agentic environments such as Google Antigravity, which coordinates local generative nodes and digital content creation software via the Model Context Protocol (MCP).

---

## 1. Generative 3D Foundations and Hardware Budgeting

The core technical constraint in consumer-tier local asset generation is the memory ceiling of the graphics processing unit. On an NVIDIA GeForce RTX 3080 Ti equipped with 12 GB of VRAM, running state-of-the-art feedforward 3D diffusion and transformer architectures alongside image models requires disciplined model selection, quantisation, and memory lifecycle management.

Open-source 3D generation is dominated by two primary architectural families: Microsoft TRELLIS and Tencent Hunyuan3D. TRELLIS utilises native 3D Variational Autoencoders combined with large-scale spatial compression to synthesise structured representations, generating neural radiance fields, Gaussian splats, and extractable textured polygon meshes. While stock TRELLIS distributions demand between 16 GB and 24 GB of VRAM during unoptimised inference, community-developed implementations and quantised GGUF variants reduce peak memory usage to approximately 11 GB, making execution viable on 12 GB hardware.

Conversely, Tencent's Hunyuan3D-2 framework employs a bifurcated DiT architecture that naturally accommodates 12 GB limits. The shape generation module can execute within 6 GB of VRAM (and as low as 5 GB when utilising the lightweight Mini variant), while the texture synthesis pass runs sequentially, allowing prior model weights to be flushed from VRAM before texture generation commences.

| Framework and Variant | Native VRAM Requirement | Optimised 12 GB Execution Strategy | Output Representation | Average Generation Time (RTX 3080 Ti) |
| :--- | :--- | :--- | :--- | :--- |
| **Microsoft TRELLIS (Base)** | 16 GB | FP16 precision, memory-managed loaders, lazy offloading | Radiance field, Gaussian splat, polygon mesh | 40 to 90 seconds |
| **TRELLIS-2 (GGUF / Quantised)** | 24 GB | GGUF quantisation, DINOv3 `torch.no_grad()` patches, model cache unregistration | High-density mesh, preliminary PBR maps, GLB | 3 to 7 minutes |
| **Tencent Hunyuan3D-2 (Standard)** | 12 GB | Sequential node execution (Shape generation → VRAM purge → Texture baking) | Raw mesh geometry, diffuse texture map | 60 to 120 seconds |
| **Hunyuan3D-2mini** | 5 GB | Standard execution with headroom for concurrent ComfyUI tasks | Medium-density base mesh, diffuse map | 20 to 45 seconds |

Running an image generator, a 3D synthesis model, and Blender concurrently on a 12 GB card risks triggering out-of-memory errors or system fallback to slow shared system memory. Maintaining stability requires launching ComfyUI with explicit memory management flags, such as `--medvram` or `--lowvram`, which enforce aggressive offloading of model weights to system RAM between processing steps.

Pipelines must also decouple 2D orthographic concept generation from 3D extraction, writing intermediate files to disk and clearing GPU caches before invoking 3D generation nodes. Furthermore, critical software patches must be implemented in local custom node environments, such as wrapping vision feature extractors in `torch.no_grad()` calls to prevent multi-gigabyte memory leaks over iterative runs.

---

## 2. Photometric Integrity: De-Lighting, Texture Projection, and PBR Decomposition

The visual flaw most responsible for low-quality AI assets is improper surface shading. Standard 2D image generators synthesise assets with baked-in directional lighting, cast shadows, and specular hot-spots derived from their training data. When an asset textured with baked lighting is imported into a real-time game engine, these pre-existing shadows conflict with dynamic in-engine lighting, causing objects to appear visually disconnected from their environment.

Achieving commercial fidelity requires complete photometric de-lighting to isolate the intrinsic base colour (albedo) of the material before generating physical surface properties. In a professional ComfyUI pipeline, the raw concept or multi-view orthographic render is routed through specialised de-lighting models such as `ComfyUI_StableDelight_ll` or the frequency-separating algorithms of `MKRShift x1TextureDelight`.

These nodes analyse luminance gradients and shadow contours, lifting occluded regions and compressing specular highlights to reconstruct a flat, neutral albedo that responds correctly to real-time microfacet BRDF shaders.

Once the albedo map is isolated, the asset must be decomposed into a physically based rendering (PBR) texture set. Neural estimation models such as Marigold or DSINE predict geometric surface normals and high-precision depth fields directly from the de-lit plate.

These depth and normal passes are processed through nodes within extensions like `ComfyUI-TextureAlchemy` to extract surface roughness, identify metallic surfaces, and approximate contact ambient occlusion, packing them into an industry-standard ORM (Ambient Occlusion, Roughness, Metallic) texture map for runtime engines.

| PBR Material Channel | Derivation Method | Standard Data Format | Functional Engine Role |
| :--- | :--- | :--- | :--- |
| **Base Colour / Albedo** | Photometric de-lighting via StableDelight or MKRShift frequency flattening | Linear RGB (read as sRGB), shadow- and highlight-free | Provides pure surface colour without interacting with engine light calculations. |
| **Normal Map** | Neural estimation via Marigold or baked from high-poly sculpts | Tangent Space, 16-bit OpenGL (`Y+`) or DirectX (`Y-`) | Perturbs surface normals to simulate high-frequency geometric detail without polygon costs. |
| **Roughness** | Luminance extraction, micro-structure variation, and material masks | Greyscale scalar (`0.0` = polished, `1.0` = fully diffuse) | Controls the specular reflection distribution and highlight tightness. |
| **Metallic** | Semantic masking or thresholded conductivity maps | Binary greyscale (`0` = dielectric, `1` = conductive metal) | Determines whether specular reflections are tinted by the albedo or left clear. |
| **Ambient Occlusion (AO)** | Depth/normal cavity integration or baked ray-distance occlusion | Greyscale scalar, packed into ORM Channel Red | Simulates soft environmental shadow attenuation in recessed crevices. |

Texturing 3D geometry from a single 2D projection inevitably leads to severe pixel stretching along perpendicular surfaces. Modern pipelines resolve this through multi-view projection texturing using software such as StableProjectorz or node-based multi-camera projection graphs.

The target 3D mesh is viewed through multiple virtual cameras (typically front, back, lateral, and overhead perspectives) that render out depth and normal maps. These renders serve as conditioning inputs for ControlNet-driven diffusion inside ComfyUI, generating angle-specific texture projections.

The software then re-projects these generations onto the shared UV layout of the mesh, using depth testing and angle falloff to blend overlapping seams while eliminating occluded camera angles.

---

## 3. Mesh Remediation: Topology, UVs, and Headless Automation

Direct generative 3D outputs consist of isotropic, unstructured triangular meshes generated through volumetric marching cubes or surface reconstruction algorithms. These raw meshes are unsuitable for game engines due to excessive vertex density, non-manifold boundaries, self-intersecting faces, and fragmented UV unwraps that waste texture resolution.

Commercial pipelines resolve these defects by using Blender as an automated, headless geometry processing engine. Rather than manually performing repetitive cleanup, developers execute modular Python scripts using Blender's command-line interface (`blender.exe --background --python wrapper.py`), passing asset paths and processing configurations via operating system environment variables or CLI arguments.

The automated cleanup sequence begins with geometric validation. The script imports the raw GLTF or OBJ asset, welds vertices within a minute spatial distance to seal open cracks, dissolves degenerate zero-area faces, and recalculates all face normal vectors to ensure uniform outward orientation.

Once topological continuity is established, the mesh undergoes retopology. For static environmental props, automated decimation algorithms collapse planar regions while preserving silhouette boundaries, reducing triangle counts from hundreds of thousands to optimised budgets between 2,000 and 10,000 polygons. For deformable characters, the pipeline triggers automated quad-remeshing algorithms that generate structured loops around areas of deformation.

Following retopology, the headless script generates clean UV coordinates. Seams are automatically marked along sharp edge angles (typically between 45 and 66 degrees) and along non-visible concavities.

The islands are unwrapped using conformal angle-based algorithms and packed with margin spacing to prevent mipmap texture bleeding. Finally, the script executes a high-to-low surface baking pass, projecting geometric normal vectors, ambient occlusion, and material maps from the unoptimised generative model onto the newly unwrapped low-poly game asset.

---

## 4. Character Rigging and Natural Locomotion

Unnatural movement in AI-generated characters typically stems from two core issues: collapsed joint geometry caused by poor skeletal weight painting, and procedural foot-sliding caused by animations that lack physical mass, momentum, and root-motion tracking.

The solution requires standardising character generation upstream. Using ControlNet OpenPose within ComfyUI, developers generate concepts in consistent, standardised A-poses or T-poses on flat backgrounds. Standardising the initial pose ensures anatomical landmarks fall within predictable spatial bounds, enabling automated rigging systems to operate reliably.

Reallusion AccuRIG has become an industry standard for automating character skeletal structures. The tool scans the incoming mesh, locates anatomical landmarks (such as the groin, knees, ankles, wrists, and neck), and constructs a standardised skeletal hierarchy.

Its multi-digit detection allows automatic rigging of hands with varying finger counts, computing smooth skin-weights that prevent limbs from collapsing or twisting unnaturally during deformation. When imported into Blender, the armature can be upgraded to a full Rigify rig, providing animators with standard Inverse Kinematics (IK) and Forward Kinematics (FK) controls.

To achieve natural movement without manual frame-by-frame animation, developers increasingly run character rigs through Cascadeur, a dedicated 3D animation environment that integrates neural-network posing with physical solvers.

Cascadeur's AutoPosing uses deep neural networks to infer natural full-body postures from the placement of a few primary control joints, dramatically reducing the time needed to block out locomotion cycles.

Once key poses are established, the AutoPhysics system analyses the character's movement across time against real-world physical dynamics. By defining fulcrum points where the character contacts the environment (such as the ball of the foot during a stride or a hand on a ledge), the physics engine computes centre-of-mass balancing, momentum conservation, and ballistic trajectories.

This eliminates the sliding feet and floating centre-of-mass common in generative video or standard motion-capture data, producing grounded, weight-responsive animations. In addition, secondary motion sliders introduce dynamic inertia to extremities and equipment, resolving physical follow-through without runtime physics overhead.

| Pipeline Stage | Dedicated Tool | Input Data | Output Data | Primary Problem Resolved |
| :--- | :--- | :--- | :--- | :--- |
| **Skeletal Fitting** | Reallusion AccuRIG | Retopologised A-Pose / T-Pose Mesh | Skeletal hierarchy and vertex weights (FBX) | Eliminates manual bone placement and joint deformation collapsing. |
| **Control Rigging** | Blender (Rigify via CC Addon) | AccuRIG Skeletal FBX | Comprehensive IK/FK animator rig | Provides tactile animator handles and constraint-driven manipulation. |
| **Pose Synthesis** | Cascadeur (AutoPosing) | Skeletal base poses | Anatomically plausible keyframes | Prevents unnatural joint angles and speeds up animation blocking. |
| **Physics Correction** | Cascadeur (AutoPhysics) | Unrefined keyframe animation | Trajectory- and mass-corrected motion curves | Eliminates foot-skating, floating centre-of-mass, and non-ballistic jumps. |
| **Engine Runtime** | Target Engine (Unreal, Unity, Godot) | Baked skeletal animation clips | Runtime state machine locomotion | Ensures responsive transitions and root-motion synchronisation in-game. |

---

## 5. Agentic Studio Automation: Google Antigravity and the Blender MCP Integration

Google Antigravity functions as an agentic development environment capable of orchestrating complex production workflows. Antigravity deploys multimodal models configured with agentic execution loops to interact directly with local software environments.

The primary integration mechanism connecting Antigravity to 3D production is the Model Context Protocol (MCP). Through an MCP server (`mcp-for-blender` or `blender-mcp-antigravity`), Antigravity establishes a bi-directional communication bridge with an active Blender instance via a local socket connection (typically port 9876).

This setup allows the agent to execute native Blender Python (`bpy`) commands, inspect the scene hierarchy, modify geometry, apply materials, and trigger rendering passes programmatically.

By leveraging this MCP interface, Antigravity functions as an automated technical director through closed-loop visual critique. When tasked with creating an environment or processing an asset, Antigravity can assemble modular architectural components, place assets sourced from Poly Haven or Sketchfab, and configure physical lighting setups.

The agent then calls viewport capture tools to render intermediate camera frames, which are returned directly to its multimodal context window.

Antigravity inspects these renders for balance, silhouette clarity, and material response, autonomously writing corrective scripts to adjust light intensity, tweak roughness values, or refine camera perspectives.

Finally, the agent can trigger decimation, texture baking, and file export operations to prepare game-ready assets for the target engine, significantly reducing the manual overhead involved in preparing generative content.

---

## 6. The Recommended Production Workflow for 12 GB VRAM Systems

To consistently generate high-fidelity, game-ready assets on an NVIDIA GeForce RTX 3080 Ti (12 GB) without exhausting available VRAM, developers should adopt a decoupled, five-phase production pipeline.

### Phase 1: Orthographic Concepting and Photometric De-Lighting (ComfyUI)
The process begins in ComfyUI by generating a clean orthographic concept sheet. Using FLUX.1-dev or a tuned SDXL checkpoint conditioned by ControlNet OpenPose, the developer generates three-view reference imagery (Front, Profile, Back) locked in an A-pose on a neutral, solid grey background.

The output is immediately routed into a de-lighting node (`ComfyUI_StableDelight_ll` or `MKRShift x1TextureDelight`), stripping away cast shadows and specular reflections to yield a flat diffuse albedo. This de-lit image is written directly to disk, and ComfyUI's model memory is cleared to free VRAM for downstream generation.

### Phase 2: Feedforward 3D Geometric Synthesis (ComfyUI / Standalone DiT)
The de-lit concept image is fed into a memory-optimised 3D generation model. On a 12 GB system, developers can run Tencent Hunyuan3D-2 using its sequential execution mode, generating the structural shape with an octree resolution of 256 to 384 before proceeding to the texturing pass.

Alternatively, a community-quantised GGUF build of Microsoft TRELLIS can be used to generate the mesh geometry. Once the raw geometric representation is generated, it is exported as an unoptimised OBJ or GLB file into a dedicated project staging directory.

### Phase 3: Mesh Remediation, Retopology, and UV Baking (Antigravity and Blender MCP)
With the raw mesh staged, the developer opens Blender with the Blender MCP server active. Using Google Antigravity, the developer instructs the agent to execute an automated mesh remediation pass.

The agent runs a Python script that seals non-manifold geometry, aligns the mesh origin to the ground plane, and applies quad-remeshing to characters or decimation to hard-surface objects.

The script then generates angle-based UV seams, packs the UV layout, and bakes high-poly normal, ambient occlusion, and cavity maps onto the low-poly mesh.

Antigravity requests an internal viewport render, verifies the bake quality, and exports the remediated, textured asset as an engine-ready FBX or GLB.

### Phase 4: Biomechanical Articulation and Physics Refinement (AccuRIG and Cascadeur)
For character assets, the retopologised mesh is imported into Reallusion AccuRIG. The software's automatic landmark solver places the humanoid skeleton, allows quick verification of finger and joint placements, and computes smooth vertex deformation weights.

The resulting armature is exported to Cascadeur, where key locomotion poses (such as walk, run, or attack cycles) are blocked out using the AI AutoPosing system.

The developer then activates AutoPhysics, sets fulcrum points on the ground, and lets the software resolve realistic balance, inertia, and ballistic curves.

Secondary motion is adjusted to add weight to clothing and hair, and the final physics-corrected animation clips are baked and exported as FBX files.

### Phase 5: Runtime Engine Integration
The final assets are imported into the target game engine (Unreal Engine 5, Unity, or Godot).

Materials are configured by connecting the de-lit Albedo map alongside the baked Normal and packed ORM (Ambient Occlusion, Roughness, Metallic) maps, ensuring clean physical interaction with the engine's dynamic lighting.

Antigravity is then used to write the gameplay code, animation state machines, and character controllers, binding the player's inputs directly to the physics-corrected animation sequences to achieve responsive, grounded gameplay.

---

## 7. Conclusion

The gap between unpresentable AI hobbyist experiments and polished, commercial AI-assisted games is fundamentally an architectural one. Successful developers avoid expecting a single generative model to produce game-ready assets.

Instead, they treat generative tools as modular accelerators within a rigorous technical pipeline: ComfyUI isolates 2D concepts and removes environmental lighting; local 3D models like Hunyuan3D-2 and quantised TRELLIS synthesise raw geometry within strict 12 GB VRAM constraints; Google Antigravity and Blender MCP automate mesh remediation, retopology, and texture baking; and AccuRIG and Cascadeur ensure characters move with correct anatomical weights and physical inertia.

By adopting this decoupled approach, developers can overcome the hardware limits of an RTX 3080 Ti and reliably produce high-quality, performant 3D assets for modern real-time game engines.

---

## 8. Comparative Analysis: How `VideoIntoBlenderModel` (`v2bm`) Maps to This Pipeline

The table below maps each stage of the industry AI game asset pipeline against our current implementation in [`VideoIntoBlenderModel`](file:///e:/Development/VideoIntoBlenderModel/AGENTS.md) and identifies high-leverage architectural improvements.

| Industry Pipeline Domain | Research Article Approach | What `v2bm` Already Implements | What `v2bm` Does Differently (Advantages) | Gaps & High-Impact Improvements for `v2bm` |
| :--- | :--- | :--- | :--- | :--- |
| **1. Hardware & VRAM Budgeting (12 GB)** | Decoupled ComfyUI stages (`--medvram`), TRELLIS-2 GGUF / Hunyuan3D-2 sequential execution, intermediate disk caching. | Zero-PyTorch runtime in `v2bm` (~20 MB MediaPipe Apache-2.0 weights + CuPy CUDA acceleration in [`v2bm/cuda_accel.py`](file:///e:/Development/VideoIntoBlenderModel/v2bm/cuda_accel.py)) plus cached `skeleton_motion.npz`. | `v2bm` consumes < 1.5 GB VRAM during motion extraction/skinning, leaving headroom on 12 GB GPUs, and [`docs/comfyui-autorigging-plan.md`](file:///e:/Development/VideoIntoBlenderModel/docs/comfyui-autorigging-plan.md) already integrates TRELLIS-2 + `MoGe-2` + `BiRefNet`. | Add explicit ComfyUI VRAM purge hooks and photometric de-lighting (`StableDelight` / `MKRShift`) to our ComfyUI character generation workflow before `TRELLIS.2` / `BakeTextureFromVoxel`. |
| **2. Photometric De-Lighting & Full PBR (Albedo + Normal + ORM)** | De-light albedo in ComfyUI; derive Normal (`Marigold`/`DSINE` or high-to-low bake) and packed ORM (Occlusion, Roughness, Metallic). | [`v2bm/rig/skinned_mesh.py`](file:///e:/Development/VideoIntoBlenderModel/v2bm/rig/skinned_mesh.py) and [`v2bm/rig/gltf.py`](file:///e:/Development/VideoIntoBlenderModel/v2bm/rig/gltf.py) extract and embed only a single `baseColorTexture` (`diffuse.png`) with scalar constants (`metallic=0.05`, `roughness=0.35`). | Fast single-texture pipeline with alpha-card preservation for hair/cape decals. | **Major Gap**: `gltf.py`, `skinned_mesh.py`, and `prepare.py` currently discard Normal, Roughness, Metallic, and AO maps baked by ComfyUI (`Node 224` & `Node 233`). Extending `.glb` export to carry `normalTexture`, `metallicRoughnessTexture` (ORM), and `occlusionTexture` + optional frequency-separation de-lighting in `prepare-rig` will dramatically elevate in-engine shading. |
| **3. Mesh Remediation & Retopology** | Headless Blender (`--background --python`): vertex welding, non-manifold repair, quad-remesh / decimate, 66° UV unwrap, high-to-low baking. | [`v2bm/rig/prepare.py`](file:///e:/Development/VideoIntoBlenderModel/v2bm/rig/prepare.py) runs headless Blender to weld UV seams, prune marching-cubes topological webbing ([`prune_topological_webbing_in_blender`](file:///e:/Development/VideoIntoBlenderModel/v2bm/rig/prepare.py#L606-L790)), apply facial-protected UV-delimited decimation (>80k -> 64k), and run `smart_project(angle_limit=66.0)`. | `v2bm` goes further on generative defects: it specifically detects and removes cross-limb bridging webbing, floating marching-cubes crumbs (<35 faces), and distal skin-weight bleeding across contacting limbs, while protecting anterior facial density for 52 ARKit blendshapes. | Add optional **Quad-Remeshing** (`quadriflow_remesh`) with high-to-low normal/albedo re-baking for raw unrigged meshes, and expose game-engine polygon budget tiers (e.g. `--poly-budget 10k / 25k / 64k`). |
| **4. Character Rigging & Facial Articulation** | Reallusion AccuRIG (manual GUI) -> Blender Rigify upgrade. | Automated 61-bone canonical humanoid conversion + 30 phalangeal finger bones + **52 Apple ARKit facial blendshapes** in [`v2bm/rig/prepare.py`](file:///e:/Development/VideoIntoBlenderModel/v2bm/rig/prepare.py), plus planned native `BVHTree` ray-cast auto-rigger ([`docs/auto-rigging-roadmap.md`](file:///e:/Development/VideoIntoBlenderModel/docs/auto-rigging-roadmap.md)). | Fully headless, 100% commercial-permissive CLI (`v2bm prepare-rig`) with zero manual GUI clicks, plus full 52-ARKit facial morph target synthesis and monocular video facial/hand capture (which AccuRIG + Cascadeur do not extract from video). | Complete the native unrigged `.glb`/`.obj` auto-rigger (`v2bm/rig/autorig.py`) and optional **Blender Rigify control rig generation** for animators who want to hand-edit animations in Blender after `v2bm` extraction. |
| **5. Natural Locomotion & Physics-Informed Kinematics** | Cascadeur AutoPosing + AutoPhysics (fulcrum contact points, CoM balancing, ballistic jump curves, secondary inertia). | Monocular video/photo extraction ([`v2bm/extract.py`](file:///e:/Development/VideoIntoBlenderModel/v2bm/extract.py)), sliding-window least-squares IK trajectory optimiser ([`v2bm/ik_optimiser.py`](file:///e:/Development/VideoIntoBlenderModel/v2bm/ik_optimiser.py)), analytical 2-bone Foot IK & stance pinning (<1 mm skate) ([`v2bm/rig/foot_ik.py`](file:///e:/Development/VideoIntoBlenderModel/v2bm/rig/foot_ik.py)), and airborne lift detection ([`v2bm/rig/animation.py`](file:///e:/Development/VideoIntoBlenderModel/v2bm/rig/animation.py)). | Extracts real human performances directly from monocular video or stills (body + 30-bone hands + 52-ARKit face) without manual keyframing. | **High-Value Physics Upgrades**:<br>1. **Ballistic Parabola Fitting** during aerial intervals (`flight_intervals` in [`_compute_airborne_lift_m`](file:///e:/Development/VideoIntoBlenderModel/v2bm/rig/animation.py#L93-L225)) enforcing $y(t) = y_0 + v_{y0} t - \frac{1}{2} g t^2$.<br>2. **Zero-Moment Point (ZMP) / Centre-of-Mass (CoM) Projection** over the support polygon during single/double-foot stance.<br>3. **Upper-Body Fulcrum Pinning** (hands on floor/ledges during breakdancing or push-ups).<br>4. **Procedural Secondary Motion / Spring Bones** for hair, capes, and equipment. |
| **6. Agentic Studio & Engine Integration** | Antigravity + Blender MCP closed-loop visual critique -> Unreal / Unity / Godot state machines. | Headless Blender studio renderer ([`scripts/render_blender_animation.py`](file:///e:/Development/VideoIntoBlenderModel/scripts/render_blender_animation.py)), Blender 4.2+ Extension ([`blender_extension/v2bm_blender/`](file:///e:/Development/VideoIntoBlenderModel/blender_extension/v2bm_blender)), and single-file `.glb` export. | Native Blender 4.2+ Extension (`v2bm_blender`) with live progress streaming and automated multi-view comparison videos (`compare.mp4`). | Add **Root-Motion Extraction Mode** (`--root-motion` separating horizontal capsule locomotion onto a ground root track for Godot/Unity/UE5 state machines) and **Loop Closure Blending** (`--loop` for walk/idle game cycles). |
