using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using NeuralTerrainGeneration;

namespace NeuralTerrainGeneration
{
    public class NeighborBlender
    {
        private TensorMathHelper tensorMathHelper = new TensorMathHelper();
        private TerrainHelper terrainHelper = new TerrainHelper();

        public void BlendSingleNeighbor(
            Terrain neighbor, 
            Tensor mirror, 
            Tensor gradient, 
            int xOffset, 
            int yOffset,
            int terrainWidth,
            int terrainHeight,
            bool keepNeighborHeights = false
        )
        {
            Tensor localGradient = new Tensor(1, terrainHeight, terrainWidth, 1);
            for(int x = 0; x < terrainWidth; x++)
            {
                for(int y = 0; y < terrainHeight; y++)
                {
                    localGradient[0, x, y, 0] = gradient[0, yOffset + y, xOffset + x, 0];
                }
            }
            Tensor scaledMirror = tensorMathHelper.Mul(localGradient, mirror);

            if(keepNeighborHeights)
            {
                float[,] neighborHeightmapArray = 
                    neighbor.terrainData.GetHeights(0, 0, terrainWidth, terrainHeight);
                Tensor neighborHeightmap = 
                    tensorMathHelper.TwoDimArrToTensor(neighborHeightmapArray);
                Tensor OnesTensor = 
                    tensorMathHelper.Populated(1.0f, terrainWidth, terrainHeight);
                Tensor inverseLocalGradient = 
                    tensorMathHelper.Sub(OnesTensor, localGradient);
                Tensor scaledNeighbor = 
                    tensorMathHelper.Mul(inverseLocalGradient, neighborHeightmap);
                Tensor blended = tensorMathHelper.Add(scaledMirror, scaledNeighbor);

                terrainHelper.SetTerrainHeights(
                    neighbor, 
                    blended.ToReadOnlyArray(), 
                    terrainWidth, 
                    terrainHeight, 
                    1, 
                    false
                );

                neighborHeightmap.Dispose();
                OnesTensor.Dispose();
                inverseLocalGradient.Dispose();
                scaledNeighbor.Dispose();
                blended.Dispose();
            }
            else
            {
                terrainHelper.SetTerrainHeights(
                    neighbor, 
                    scaledMirror.ToReadOnlyArray(), 
                    terrainWidth, 
                    terrainHeight, 
                    1, 
                    false
                );
            }
    
            localGradient.Dispose();
            scaledMirror.Dispose();
        }

        public void ClampToSingleNeighbor(
            Terrain terrain,
            float[,] heightmap, 
            Terrain neighbor, 
            int ownOffset,
            int neighborOffset,
            bool isHorizontalNeighbor,
            int terrainWidth,
            int terrainHeight
        )
        {
            if(neighbor != null)
            {
                float[,] neighborHeightmap = neighbor.terrainData.GetHeights(
                    0, 0, terrainWidth, terrainHeight
                );

                if(isHorizontalNeighbor)
                {
                    // If isHorizontalNeighbor, clamp along the y axis.
                    for(int i = 0; i < terrainHeight; i++)
                    {
                        heightmap[i, ownOffset] = neighborHeightmap[i, neighborOffset];
                    }
                }
                else
                {
                    // if !isHorizontalNeighbor, clamp along the x axis.
                    for(int i = 0; i < terrainWidth; i++)
                    {
                        heightmap[ownOffset, i] = neighborHeightmap[neighborOffset, i];
                    }
                }
            }

            Tensor heightmapTensor = tensorMathHelper.TwoDimArrToTensor(heightmap);
            terrainHelper.SetTerrainHeights(
                terrain, 
                heightmapTensor.ToReadOnlyArray(), 
                terrainWidth, 
                terrainHeight, 
                1, 
                false
            );
            heightmapTensor.Dispose();
        }

        public void ClampToNeighbors(
            Terrain terrain, 
            Terrain left, 
            Terrain right, 
            Terrain top, 
            Terrain bottom, 
            int width, 
            int height
        )
        {
            float[,] heightmap = 
                terrain.terrainData.GetHeights(0, 0, width, width);

            // Clamp to left neighbor.
            ClampToSingleNeighbor(
                terrain, heightmap, left, 0, width-1, true, width, height
            );

            // Clamp to right neighbor.
            ClampToSingleNeighbor(
                terrain, heightmap, right, width-1, 0, true, width, height
            );

            // Clamp to top neighbor.
            ClampToSingleNeighbor(
                terrain, heightmap, top, height-1, 0, false, width, height
            );

            // Clamp to bottom neighbor.
            ClampToSingleNeighbor(
                terrain, heightmap, bottom, 0, height-1, false, width, height
            );

            terrain.terrainData.SetHeights(0, 0, heightmap);
        }

        public void BlendAllNeighbors(
            Terrain terrain, 
            int width, 
            int height, 
            float radius1, 
            float radius2, 
            float bValue,
            bool keepNeighborHeights = false
        )
        {
            TensorMathHelper tensorMathHelper = new TensorMathHelper();
            float[,] heightmap = 
                terrain.terrainData.GetHeights(0, 0, width, height);
            Tensor heightmapTensor = 
                tensorMathHelper.TwoDimArrToTensor(heightmap);
            Tensor horizontalMirror = 
                tensorMathHelper.Mirror(heightmapTensor, false, true);
            Tensor verticalMirror = 
                tensorMathHelper.Mirror(heightmapTensor, true, false);
            Tensor bothMirror = 
                tensorMathHelper.Mirror(heightmapTensor, true, true);

            Tensor gradient = new Tensor(1, width * 3, height * 3, 1);
            Vector2 center = new Vector2(radius1 + radius2, radius1 + radius2);
            for(int x = 0; x < width * 3; x++)
            {
                for(int y = 0; y < height * 3; y++)
                {
                    float distance = Vector2.Distance(new Vector2(x, y), center);
                    if(distance < radius1)
                    {
                        gradient[0, x, y, 0] = 1.0f;
                    }
                    else
                    {
                        float gradientValue = (-1.0f / radius1) * distance + bValue;
                        if(gradientValue > 1.0f)
                        {
                            gradient[0, x, y, 0] = 1.0f;
                        }
                        else
                        {
                            gradient[0, x, y, 0] = gradientValue;
                        }
                    }
                }
            }
            
            Terrain topLeft = null;
            Terrain bottomLeft = null;
            Terrain topRight = null;
            Terrain bottomRight = null;

            // Split gradient in 8 parts.
            Terrain left = terrain.leftNeighbor;
            if(left != null)
            {
                BlendSingleNeighbor(
                    left, 
                    horizontalMirror, 
                    gradient, 
                    0, 
                    height,
                    width,
                    height,
                    keepNeighborHeights
                );

                topLeft = left.topNeighbor;
                bottomLeft = left.bottomNeighbor;
            }

            Terrain right = terrain.rightNeighbor;
            if(right != null)
            {
                BlendSingleNeighbor(
                    right, 
                    horizontalMirror, 
                    gradient, 
                    width * 2, 
                    height, 
                    width, 
                    height,
                    keepNeighborHeights
                );

                topRight = right.topNeighbor;
                bottomRight = right.bottomNeighbor;
            }

            Terrain top = terrain.topNeighbor;
            if(top != null)
            {
                BlendSingleNeighbor(
                    top, 
                    verticalMirror, 
                    gradient, 
                    width, 
                    height * 2, 
                    width, 
                    height,
                    keepNeighborHeights
                );
            }

            Terrain bottom = terrain.bottomNeighbor;
            if(bottom != null)
            {
                BlendSingleNeighbor(
                    bottom, 
                    verticalMirror, 
                    gradient, 
                    width, 
                    0, 
                    width, 
                    height,
                    keepNeighborHeights
                );
            }
            
            if(topLeft != null)
            {
                BlendSingleNeighbor(
                    topLeft, 
                    bothMirror, 
                    gradient, 
                    0, 
                    height * 2, 
                    width, 
                    height,
                    keepNeighborHeights
                );

                ClampToNeighbors(
                    topLeft,
                    null,
                    top,
                    null,
                    left,
                    width,
                    height
                );
            }

            if(bottomLeft != null)
            {
                BlendSingleNeighbor(
                    bottomLeft, 
                    bothMirror, 
                    gradient, 
                    0, 
                    0, 
                    width, 
                    height,
                    keepNeighborHeights
                );

                ClampToNeighbors(
                    bottomLeft,
                    null,
                    bottom,
                    left,
                    null,
                    width,
                    height
                );
            }

            if(topRight != null)
            {
                BlendSingleNeighbor(
                    topRight, 
                    bothMirror, 
                    gradient, 
                    width * 2, 
                    height * 2, 
                    width, 
                    height,
                    keepNeighborHeights
                );

                ClampToNeighbors(
                    topRight,
                    top,
                    null,
                    null,
                    right,
                    width,
                    height
                );
            }

            if(bottomRight != null)
            {
                BlendSingleNeighbor(
                    bottomRight, 
                    bothMirror, 
                    gradient, 
                    width * 2, 
                    0, 
                    width, 
                    height,
                    keepNeighborHeights
                );

                ClampToNeighbors(
                    bottomRight,
                    bottom,
                    null,
                    right,
                    null,
                    width,
                    height
                );
            }
        }
    }
}
