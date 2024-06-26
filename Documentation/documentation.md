---
fontfamily: charter
---

# Neural Terrain Generation 1.2.1 Documentation

--- 

# Getting Started
1. Create a terrain object
2. Open to terrain tools dropdown from the second tab of the terrain component
3. Select "Neural Terrain Generation"
4. Click "Generate Terrain From Scratch"

# Neural Terrain Generation UI Guide

## Model Asset 
The model that will be used to generate terrain. Currently only pix_diffuser_epoch62.onnx is available.

## Worker Type
The backend that will run your chosen model:

* CPU

    * CSharpBurst: highly efficient, jobified and parallelized CPU code compiled via Burst.
    * CSharp: slightly less efficient CPU code.
    * CSharpRef: a less efficient but more stable reference implementation.

* GPU

    * ComputePrecompiled: highly efficient GPU code with all overhead code stripped away and precompiled into the worker. (Recommended)
    * Compute: highly efficient GPU but with some logic overhead.
    * ComputeRef: a less efficient but more stable reference implementation.

These will each give varying performance depending on your hardware. If you are unsure, try ComputePrecompiled first, and if that doesn't work very well, try CSharpBurst.

## Height Multiplier
Multiplier applied to heightmaps to scale them up or down.

## Sampling Steps
The number of reverse diffusion steps that will be used to generate heightmaps. Higher values will take longer to compute. 

## UpSampler Type
The method used to upsample the output of the model:

* Barracuda: fast bilinear upsampling using the Barracuda library.
* Custom: slower custom implementation of bicubic upsampling.

## UpSample Resolution
The resolution that the model's output will be upsampled to:

* 256
* 512
* 1024
* 2048
* 4096

Because the models's base output is 256x256, no upsampling will occur if this is set to 256.

## Smoothing Enabled
Determines whether or not the output of the model will be automatically smoothed. This is recommended as it will remove the blocky artifacts that are common in the output of the model. When enabled it will also reveal the smoothing options kernel size and sigma.

## Kernel Size
The size of the kernel used to smooth the output of the model. It corresponds to the kernel of a Gaussian blur. A larger kernel will result in more smoothing, but will also take longer to compute.

## Sigma
The standard deviation of the Gaussian blur used to smooth the output of the model.

## Random Seed
Determines whether or not a random seed will be used when generating terrain.

## Seed
When random seed is enabled, this value is the seed that was used to generate the last terrain. Conversely, when random seed is disabled, this value is the seed that will be used to generate the next terrain.

## Enable Brush
Reveals brush UI.

## Disable Brush
Hides brush UI.

## Brush Mask
Mask applied to brush heightmap in order to blend out hard edges and control its shape. All brush masks must be 256x256. Default masks can be found in NeuralTerrainGeneration/BrushMasks.

## Opacity
Opacity of the brush.

## Size
Size of the brush.

## Rotation
Rotation of the brush.

## Height offset
Value subtracted from the brush heightmap. This is used to sink the brush into the terrain for better blending.

## Stamp Mode
When this is enabled, the brush will be discretely stamped onto the terrain instead of continuously painted, in other words, you will only be able to place one brush stroke per click.

## Generate Brush Heightmap
Generates a brush heightmap using the chosen parameters.

## Generate Terrain From Scratch
Generates a heightmap from scratch using the chosen parameters, and automatically applies it to the selected terrain.

## Blend Function Start Value
Controls the inner radius of the circular gradient used to blend neighboring terrains.

## Blend With Neighbors
Blends the terrain with its neighbors.