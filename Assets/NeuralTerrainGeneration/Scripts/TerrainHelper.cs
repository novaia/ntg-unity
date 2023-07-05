using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuralTerrainGeneration;

namespace NeuralTerrainGeneration
{
    public class TerrainHelper
    {
        public void SetTerrainHeights(
            Terrain terrain, 
            float[] heightmap, 
            int width, 
            int height, 
            float heightMultiplier, 
            bool scale = true
        )
        {
            terrain.terrainData.heightmapResolution = width;

            float scaleCoefficient = 1;
            if(scale)
            {
                float maxValue = heightmap[0];
                for(int i = 0; i < heightmap.Length; i++)
                {
                    if(heightmap[i] > maxValue)
                    {
                        maxValue = heightmap[i];
                    }
                }
                scaleCoefficient = (1 / maxValue) * heightMultiplier;
            }

            float[,] newHeightmap = new float[width+1, height+1];
            for(int x = 0; x < width; x++)
            {
                for(int y = 0; y < height; y++)
                {
                    newHeightmap[x, y] = heightmap[x + y * width] * scaleCoefficient;
                }
            }

            // This is to prevent falloff artifacts at the edge of the terrain.
            for(int i = 0; i < width+1; i++)
            {
                newHeightmap[i, height] = newHeightmap[i, height-1];
            }
            for(int i = 0; i < height+1; i++)
            {
                newHeightmap[width, i] = newHeightmap[width-1, i];
            }

            terrain.terrainData.SetHeights(0, 0, newHeightmap);
        }
    }
}
