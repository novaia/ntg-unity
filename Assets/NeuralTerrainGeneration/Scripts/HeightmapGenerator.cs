using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using NeuralTerrainGeneration;

namespace NeuralTerrainGeneration
{
    public class HeightmapGenerator
    {
        private TensorMathHelper tensorMathHelper = new TensorMathHelper();

        public float[] GenerateHeightmapFromScratch(
            int modelOutputWidth,
            int modelOutputHeight,
            int samplingSteps,
            int seed,
            bool smooth,
            BarraUpSampler barraUpSampler,
            GaussianSmoother gaussianSmoother,
            Diffuser diffuser
        )
        {
            const int batchSize = 1;
            const int channels = 1;
            Tensor input = tensorMathHelper.PseudoRandomNormalTensor(
                batchSize,
                modelOutputWidth,
                modelOutputHeight,
                channels,
                seed
            );

            Tensor baseHeightmap = diffuser.Execute(
                input,
                modelOutputWidth,
                modelOutputHeight,
                samplingSteps
            );

            Tensor upSampled = barraUpSampler.Execute(baseHeightmap);

            float[] finalHeightmap;
            if(smooth)
            {
                Tensor smoothed = gaussianSmoother.Execute(upSampled);
                finalHeightmap = smoothed.ToReadOnlyArray();
                smoothed.Dispose();
            }
            else
            {
                finalHeightmap = upSampled.ToReadOnlyArray();
            }

            input.Dispose();
            baseHeightmap.Dispose();
            upSampled.Dispose();
            return finalHeightmap;
        }
    }
}
