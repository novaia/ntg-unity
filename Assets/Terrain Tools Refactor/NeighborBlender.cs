using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

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
        Tensor scaledMirror = tensorMathHelper.MultiplyTensors(localGradient, mirror);

        if(keepNeighborHeights)
        {
            float[,] neighborHeightmapArray = neighbor.terrainData.GetHeights(
                0, 0, terrainWidth, terrainHeight
            );
            Tensor neighborHeightmap = tensorMathHelper.TwoDimensionalArrayToTensor(neighborHeightmapArray);
            Tensor OnesTensor = tensorMathHelper.PopulatedTensor(1.0f, terrainWidth, terrainHeight);
            Tensor inverseLocalGradient = tensorMathHelper.SubtractTensor(OnesTensor, localGradient);
            Tensor scaledNeighbor = tensorMathHelper.MultiplyTensors(inverseLocalGradient, neighborHeightmap);
            Tensor blended = tensorMathHelper.AddTensor(scaledMirror, scaledNeighbor);

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

    public void BlendAllNeighbors(
        Terrain terrain, 
        int terrainWidth, 
        int terrainHeight, 
        float radius1, 
        float radius2, 
        float bValue,
        bool keepNeighborHeights = false
    )
    {
        TensorMathHelper tensorMathHelper = new TensorMathHelper();
        float[,] heightmap = terrain.terrainData.GetHeights(0, 0, terrainWidth, terrainHeight);
        Tensor heightmapTensor = tensorMathHelper.TwoDimensionalArrayToTensor(heightmap);
        Tensor horizontalMirror = tensorMathHelper.MirrorTensor(heightmapTensor, false, true);
        Tensor verticalMirror = tensorMathHelper.MirrorTensor(heightmapTensor, true, false);
        Tensor bothMirror = tensorMathHelper.MirrorTensor(heightmapTensor, true, true);

        Tensor gradient = new Tensor(1, terrainWidth * 3, terrainHeight * 3, 1);
        Vector2 center = new Vector2(radius1 + radius2, radius1 + radius2);
        for(int x = 0; x < terrainWidth * 3; x++)
        {
            for(int y = 0; y < terrainHeight * 3; y++)
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
        
        Terrain topLeftNeighbor = null;
        Terrain bottomLeftNeighbor = null;
        Terrain topRightNeighbor = null;
        Terrain bottomRightNeighbor = null;

        // Split gradient in 8 parts.
        Terrain leftNeighbor = terrain.leftNeighbor;
        if(leftNeighbor != null)
        {
            BlendSingleNeighbor(
                leftNeighbor, 
                horizontalMirror, 
                gradient, 
                0, 
                terrainHeight,
                terrainWidth,
                terrainHeight,
                keepNeighborHeights
            );

            topLeftNeighbor = leftNeighbor.topNeighbor;
            bottomLeftNeighbor = leftNeighbor.bottomNeighbor;
        }

        Terrain rightNeighbor = terrain.rightNeighbor;
        if(rightNeighbor != null)
        {
            BlendSingleNeighbor(
                rightNeighbor, 
                horizontalMirror, 
                gradient, 
                terrainWidth * 2, 
                terrainHeight, 
                terrainWidth, 
                terrainHeight,
                keepNeighborHeights
            );

            topRightNeighbor = rightNeighbor.topNeighbor;
            bottomRightNeighbor = rightNeighbor.bottomNeighbor;
        }

        Terrain topNeighbor = terrain.topNeighbor;
        if(topNeighbor != null)
        {
            BlendSingleNeighbor(
                topNeighbor, 
                verticalMirror, 
                gradient, 
                terrainWidth, 
                terrainHeight * 2, 
                terrainWidth, 
                terrainHeight,
                keepNeighborHeights
            );
        }

        Terrain bottomNeighbor = terrain.bottomNeighbor;
        if(bottomNeighbor != null)
        {
            BlendSingleNeighbor(
                bottomNeighbor, 
                verticalMirror, 
                gradient, 
                terrainWidth, 
                0, 
                terrainWidth, 
                terrainHeight,
                keepNeighborHeights
            );
        }
        
        if(topLeftNeighbor != null)
        {
            BlendSingleNeighbor(
                topLeftNeighbor, 
                bothMirror, 
                gradient, 
                0, 
                terrainHeight * 2, 
                terrainWidth, 
                terrainHeight,
                keepNeighborHeights
            );
        }

        if(bottomLeftNeighbor != null)
        {
            BlendSingleNeighbor(
                bottomLeftNeighbor, 
                bothMirror, 
                gradient, 
                0, 
                0, 
                terrainWidth, 
                terrainHeight,
                keepNeighborHeights
            );
        }

        if(topRightNeighbor != null)
        {
            BlendSingleNeighbor(
                topRightNeighbor, 
                bothMirror, 
                gradient, 
                terrainWidth * 2, 
                terrainHeight * 2, 
                terrainWidth, 
                terrainHeight,
                keepNeighborHeights
            );
        }

        if(bottomRightNeighbor != null)
        {
            BlendSingleNeighbor(
                bottomRightNeighbor, 
                bothMirror, 
                gradient, 
                terrainWidth * 2, 
                0, 
                terrainWidth, 
                terrainHeight,
                keepNeighborHeights
            );
        }
    }

}