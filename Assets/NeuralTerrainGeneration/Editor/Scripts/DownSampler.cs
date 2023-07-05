using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

namespace NeuralTerrainGeneration
{
    public class DownSampler
    {
        public Tensor DownSample(Tensor original, int factor)
        {
            Tensor downSampled = new Tensor(
                1, original.height / factor, original.width / factor, 1
            );
            for(int x = 0; x < downSampled.width; x++)
            {
                for(int y = 0; y < downSampled.height; y++)
                {
                    downSampled[0, y, x, 0] = original[0, y * factor, x * factor, 0];
                }
            }
            return downSampled;
        }
    }
}
