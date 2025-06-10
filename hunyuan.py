from typing import Literal, Union, Dict
from chutes.chute import Chute, ChutePack, NodeSelector
from chutes.image import Image 
import time
from fastapi.responses import JSONResponse
import uuid
from pydantic import BaseModel, Field
from loguru import logger
import random
import os
from urllib.parse import urlparse
import base64

image = (
    Image(
    username="desudesuka",
    name="hunyuan3d-2",
    tag=f"{time.strftime('%Y-%m-%d')}-{int(time.time())}",
    readme="Hunyuan3D-2 to generate 3D models.",
    )
    .from_base("parachutes/python:3.12.9")
    .set_user('root')
    .run_command("apt-get update")
    .apt_install([
        "aria2"
    ])
    .set_user("chutes")
    .run_command("git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /tmp/hunyuan3d")
    .run_command("pip install -r /tmp/hunyuan3d/requirements.txt")
    .run_command("pip install -e /tmp/hunyuan3d")
    .run_command("cd /tmp/hunyuan3d/hy3dgen/texgen/custom_rasterizer && TORCH_CUDA_ARCH_LIST='6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0' pip install --no-build-isolation .")
    .run_command("cd /tmp/hunyuan3d/hy3dgen/texgen/differentiable_renderer && pip install --no-build-isolation .")
)

class InferenceRequest(BaseModel):
    image_source: Union[str, Dict[Literal['front', 'left', 'back'], str]] = Field(description="URL of the image to process, or a dictionary of URLs for front, left, and back views.")
    remove_bg: bool = Field(True, description="Whether to remove the background from the image")
    seed: int = Field(0, description="Random seed for reproducibility")
    inference_steps: int = Field(20, description="Number of inference steps to run")
    octree_resolution: int = Field(256, description="Resolution of the octree", ge=16, le=512)
    guidance_scale: float = Field(5, description="Guidance scale for the model")
    num_chunks: int = Field(1000, description="Number of chunks")
    export_type: Literal["glb", "obj", "ply", "stl"] = Field("glb", description="Type of export file")
    simplify_mesh: bool = Field(False, description="Whether to simplify the mesh")
    target_face_number: int = Field(10000, description="Target number of faces for mesh simplification", ge=100, le=1000000)

chute = Chute(
    username="desudesuka",
    name="hunyuan3d-2",
    tagline="3D Generation with Hunyuan3D-2",
    readme="Hunyuan3D-2 is a powerful model for generating 3D models from images. This chute allows you to run the model with various parameters.",
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=42
    )
)

@chute.on_startup()
async def initialize(self):
    """
    Load the classification pipeline and perform a warmup.
    """

    from PIL import Image
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import (
        Hunyuan3DDiTFlowMatchingPipeline,
        FloaterRemover,
        DegenerateFaceRemover,
        FaceReducer,
    )
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    import torch

    logger.info("Initializing Hunyuan3D-2 pipeline...")

    model_path_sv = "tencent/Hunyuan3D-2"
    self.pipeline_shapegen_sv = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path_sv
    )

    model_path_mv = 'tencent/Hunyuan3D-2mv'
    self.pipeline_shapegen_mv = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path_mv,
        subfolder='hunyuan3d-dit-v2-mv',
        variant='fp16'
    )

    self.pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path_sv)
    self.pipeline_texgen.enable_model_cpu_offload()

    self.rembg = BackgroundRemover()
    self.floater_remover = FloaterRemover()
    self.degenerate_face_remover = DegenerateFaceRemover()
    self.face_reducer = FaceReducer()
    self.export_to_trimesh = export_to_trimesh

    self.torch = torch
    self.Image = Image

    os.makedirs("tmp", exist_ok=True)
    logger.info("Hunyuan3D-2 pipeline initialized.")

MAX_SEED = int(1e7)
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def export_mesh(mesh, save_folder, textured=False, type='glb'):
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path

@chute.cord(
    public_api_path="/generate",
    method="POST",
    pass_chute=True,
    input_schema=InferenceRequest,
    stream=False,
)
async def generate(
    self,
    params: InferenceRequest,
) -> JSONResponse:
    run_id = str(uuid.uuid4())[:8]

    ( 
        seed, image_source, remove_bg, inference_steps, octree_resolution, guidance_scale, 
        num_chunks, export_type, simplify_mesh, target_face_number
    ) = (
        params.seed,
        params.image_source,
        params.remove_bg,
        params.inference_steps,
        params.octree_resolution,
        params.guidance_scale,
        params.num_chunks,
        params.export_type,
        params.simplify_mesh,
        params.target_face_number
    )
    randomize_seed = True if seed <= 0 else False

    is_multi_view = isinstance(image_source, dict)
    image_paths_to_remove = []


    if is_multi_view:
        logger.info(f"Processing multi-view request for run_id {run_id}.")
        images = {}
        for view, url in image_source.items():
            path = urlparse(url).path
            ext = os.path.splitext(path)[1]
            image_path = f"tmp/{run_id}_{view}{ext}"
            os.system(f"aria2c -x 16 {url} -o {image_path}")
            image_paths_to_remove.append(image_path)

            image = self.Image.open(image_path).convert("RGBA")
            if remove_bg:
                image = self.rembg(image)
            images[view] = image
        
        image_for_texture = images.get("front")
        if not image_for_texture:
            logger.warning("No 'front' view provided for texturing. Using the first available image.")
            image_for_texture = next(iter(images.values()))

        shape_pipeline = self.pipeline_shapegen_mv
        shape_gen_input = images
    else: # single view
        logger.info(f"Processing single-view request for run_id {run_id}.")
        path = urlparse(image_source).path
        ext = os.path.splitext(path)[1]
        image_path = f"tmp/{run_id}{ext}"
        os.system(f"aria2c -x 16 {image_source} -o {image_path}")
        image_paths_to_remove.append(image_path)
        
        image = self.Image.open(image_path).convert("RGBA")
        if remove_bg:
            image = self.rembg(image)
        
        image_for_texture = image
        shape_pipeline = self.pipeline_shapegen_sv
        shape_gen_input = image



    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = self.torch.Generator()
    generator = generator.manual_seed(int(seed))

    # 1. Generate Shape
    mesh_obj = shape_pipeline(
        image=shape_gen_input,
        num_inference_steps=inference_steps,
        octree_resolution=octree_resolution,
        guidance_scale=guidance_scale,
        num_chunks=num_chunks,
        generator=generator,
        output_type='trimesh'
    )[0]

     # 2. Convert to trimesh for cleaning and simplification
    mesh = mesh_obj
    mesh = self.floater_remover(mesh)
    mesh = self.degenerate_face_remover(mesh)

    if simplify_mesh:
        mesh = self.face_reducer(mesh, target_face_number)
    else:
        mesh = self.face_reducer(mesh)

    # 4. Generate texture on the cleaned/simplified mesh
    textured_mesh = self.pipeline_texgen(
        mesh=mesh,
        image=image_for_texture
    )

    export_type = export_type.lower()
    if export_type not in ['glb', 'obj', 'ply', 'stl']:
        logger.error(f"Invalid export type: {export_type}. Defaulting to 'glb'.")
        export_type = 'glb'

    logger.info(f"Exporting mesh as {export_type} for run_id {run_id}.")
    save_folder = f"tmp/{run_id}"
    os.makedirs(save_folder, exist_ok=True)
  
    # The output of texturing is a Hunyuan Mesh object with an export method
    mesh_path = export_mesh(textured_mesh, save_folder, textured=True, type=export_type)

    if not os.path.exists(mesh_path):
        logger.error(f"Failed to export mesh for run_id {run_id}.")
        return JSONResponse(
            content={
                "status": "error",
                "message": "Failed to export mesh."
            },
            status_code=500
        )

    logger.info(f"Generated mesh saved as {mesh_path} .")

    with open(mesh_path, "rb") as f:
        encoded_mesh = base64.b64encode(f.read()).decode('utf-8')

    # clean up
    for path in image_paths_to_remove:
        if os.path.exists(path):
            os.remove(path)
    os.remove(mesh_path)
    logger.info(f"Mesh encoded and ready for response.")
  
    response_content = {
        "status": "success",
        "run_id": run_id,
        "mesh": encoded_mesh,
        "export_type": export_type,
        "seed": seed
    }
    logger.info(f"Response content prepared for run_id {run_id}.")

    return JSONResponse(
        content=response_content,
        status_code=200
    )
